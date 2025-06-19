import os
import sys
import time
import shutil
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import cv2
import argparse
import logging
import traceback
import gc  # for garbage collection
import psutil
import platform

# figure out script dir for paths
if getattr(sys, 'frozen', False):
    # running as exe
    script_dir = os.path.dirname(sys.executable)
else:
    # regular python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

# setup log file
log_file = os.path.join(script_dir, "jpeg_processor.log")
print(f"Using log file: {log_file}")  # debug info

# setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # console output too
    ]
)

def check_folder_stability(folder_path):
    """
    Check if folder is stable - wait 5 seconds and see if it changed.
    Keeps retrying immediately until stable.
    """
    if not os.path.exists(folder_path):
        logging.warning(f"Folder does not exist: {folder_path}")
        return False
        
    while True:  # keep checking until stable
        # get initial state
        try:
            initial_files = os.listdir(folder_path)
            initial_size = sum(os.path.getsize(os.path.join(folder_path, f)) 
                            for f in initial_files if os.path.isfile(os.path.join(folder_path, f)))
            
            # wait 5 seconds
            logging.info(f"Waiting 5 seconds for folder stability: {folder_path}")
            time.sleep(5)
            
            # check if folder changed
            if not os.path.exists(folder_path):
                logging.warning(f"Folder disappeared during stability check: {folder_path}")
                return False
                
            current_files = os.listdir(folder_path)
            current_size = sum(os.path.getsize(os.path.join(folder_path, f)) 
                            for f in current_files if os.path.isfile(os.path.join(folder_path, f)))
            
            if len(initial_files) != len(current_files) or initial_size != current_size:
                # changed - immediately try again
                logging.info(f"Folder still changing, retrying stability check immediately...")
                continue
            else:
                # unchanged, stable
                logging.info(f"Folder is stable, proceeding with processing")
                return True
                
        except Exception as e:
            logging.error(f"Error checking folder stability: {str(e)}")
            return False

def parse_args():
    parser = argparse.ArgumentParser(description="JPEG Processor")
    parser.add_argument('--watch-dir', required=True, help='Directory to watch for new folders containing PDFs')
    parser.add_argument('--output-dir', required=True, help='Directory to move completed folders')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum number of worker processes (use 1 for sequential processing)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for JPEG conversion (default: 200)')
    return parser.parse_args()

def safe_remove_file(file_path, max_retries=7, retry_delay=3):
    """
    Try to delete a file, with retries if it's locked.
    Enhanced for network shares and large files.
    Returns True if the file was successfully deleted, False otherwise.
    """
    # Normalize path to handle mixed slashes in network paths
    normalized_path = os.path.normpath(file_path)
    
    # First verify the file still exists - with proper normalization
    if not os.path.exists(normalized_path):
        logging.warning(f"File does not exist for deletion (normalized): {normalized_path}")
        # Double-check with original path just in case
        if os.path.exists(file_path):
            logging.warning(f"File exists with original path but not normalized path")
            normalized_path = file_path
        else:
            return True  # Consider it a success if already gone
            
    logging.info(f"Attempting to delete file: {normalized_path}")
    
    # Wait before attempting deletion to allow file handles to release
    time.sleep(3)
    
    # Force garbage collection before attempting deletion
    gc.collect()
    
    # More aggressive memory cleanup on Windows
    if platform.system() == 'Windows':
        try:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except Exception as e:
            logging.warning(f"Error freeing memory on Windows: {str(e)}")
    
    # Now try to delete with retries
    for attempt in range(max_retries):
        try:
            # Double-check existence before each attempt
            if os.path.exists(normalized_path):
                logging.info(f"Deletion attempt {attempt+1} for: {normalized_path}")
                os.remove(normalized_path)
                
                # Verify the file is actually gone
                if not os.path.exists(normalized_path):
                    logging.info(f"Successfully removed file on attempt {attempt+1}")
                    return True
                else:
                    logging.warning(f"File still exists after remove() call - may be due to filesystem caching")
                    # Wait a bit for filesystem to update
                    time.sleep(1)
                    
                    # Check again after waiting
                    if not os.path.exists(normalized_path):
                        logging.info(f"File confirmed deleted after brief wait")
                        return True
            else:
                logging.info(f"File no longer exists, deletion successful")
                return True
        except PermissionError:
            logging.warning(f"Permission error deleting file on attempt {attempt+1}")
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} to remove file failed: {normalized_path}, Error: {str(e)}")
        
        # Use longer retry delay for later attempts
        current_delay = retry_delay * (attempt + 1)
        logging.info(f"Waiting {current_delay} seconds before retry {attempt+2}...")
        time.sleep(current_delay)
        
    # After all retries, as a last resort, try to rename the file
    try:
        logging.error(f"Failed to remove file after {max_retries} attempts: {normalized_path}")
        backup_path = normalized_path + ".processed"
        logging.info(f"Trying to rename file instead: {normalized_path} -> {backup_path}")
        os.rename(normalized_path, backup_path)
        logging.info(f"Successfully renamed file as an alternative to deletion")
        return False  # Return FALSE - the file wasn't deleted, just renamed
    except Exception as rename_error:
        logging.error(f"Failed to rename file: {str(rename_error)}")
        return False

class PDFProcessor:
    def __init__(self, watch_directory, output_directory, dpi=200):
        self.watch_directory = watch_directory
        self.output_directory = output_directory
        self.dpi = dpi
        self.stable_folders = set()  # track folders that have been verified as stable
        self.file_locks = {}  # for per-PDF locks
        self.folder_locks = {}  # for per-folder locks
        self.lock = threading.Lock()
        logging.info(f"Initializing PDFProcessor with watch dir: {watch_directory}, output dir: {output_directory}, dpi: {dpi}")

    def _process_page(self, page_data, output_path, page_num, total_pages, should_resize=True):
        """Process a single page from PDF to JPEG with optional resizing - ALL IN MEMORY"""
        try:
            # Convert page data (bytes) to numpy array for OpenCV
            nparr = np.frombuffer(page_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError(f"Failed to decode page data for page {page_num}")
            
            # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img)
            
            # Handle resizing in memory if needed (same as TIFF script)
            if should_resize:
                original_width, original_height = pil_img.size
                original_is_landscape = original_width > original_height
                
                # Set target dimensions based on orientation
                target_width = 1700
                target_height = 2200
                if original_is_landscape and target_width < target_height:
                    target_width, target_height = target_height, target_width
                elif not original_is_landscape and target_width > target_height:
                    target_width, target_height = target_height, target_width
                
                # Calculate scaling
                scale = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                
                # Resize image in memory
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Create final image with white background
                final_img = Image.new('RGB', (target_width, target_height), 'white')
                
                # Calculate centering offsets
                x_offset = (target_width - new_width) // 2
                y_offset = (target_height - new_height) // 2
                
                # Paste resized image in center
                final_img.paste(pil_img, (x_offset, y_offset))
                
                # Update img to the final resized version
                pil_img = final_img
            
            # Save directly as JPEG with DPI
            pil_img.save(output_path, "JPEG", quality=75, dpi=(self.dpi, self.dpi))
            
            # Clean up memory
            img = None
            pil_img = None
            gc.collect()
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing page {page_num}: {str(e)}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except:
                    pass
            return False

    def _process_page_with_retries(self, page_data, output_path, page_num, total_pages, should_resize=True):
        """Process a single page with infinite retries using page data (bytes)"""
        attempt = 0
        
        # infinite retry loop - never give up on a page
        while True:
            attempt += 1
            if attempt > 1:
                logging.warning(f"retry attempt {attempt} for page {page_num} of {total_pages}")
                time.sleep(1)  # brief pause before retry
            
            try:
                # process the page using page data (pass bytes directly)
                page_success = self._process_page(page_data, output_path, page_num, total_pages, should_resize)
                
                if page_success and os.path.exists(output_path):
                    # page converted successfully
                    logging.info(f"successfully converted page {page_num} of {total_pages}")
                    return True
                else:
                    logging.warning(f"failed attempt {attempt} for page {page_num}")
            except Exception as e:
                logging.error(f"error on attempt {attempt} for page {page_num}: {str(e)}")
        
        # This line should never be reached because the loop only exits with a return
        return False

    def process_pdf_to_jpgs(self, pdf_path):
        """turn pdf into jpg images using chunked processing. returns (converted files list, success)"""
        # Create file lock if it doesn't exist
        if pdf_path not in self.file_locks:
            self.file_locks[pdf_path] = threading.Lock()
            
        # Lock this PDF for exclusive access
        with self.file_locks[pdf_path]:
            converted_pages = []
            
            # Check if folder wants resizing
            folder_name = os.path.basename(os.path.dirname(pdf_path))
            should_resize = not folder_name.lower().endswith('--noresize')
            
            try:
                # Check file size to determine if memory mapping would be beneficial
                file_size = os.path.getsize(pdf_path)
                
                # Open the pdf file - try to use memory mapping if available
                try:
                    doc = fitz.open(pdf_path, filetype="pdf", memory=file_size > 50_000_000)
                except TypeError:
                    doc = fitz.open(pdf_path)
                    
                total_pages = len(doc)
                digits = len(str(total_pages)) + 1
                logging.info(f"Processing {total_pages} pages in PDF: {pdf_path}")

                # prep output paths
                pdf_dir = os.path.dirname(pdf_path)
                pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                
                # Process in chunks of 10,000 pages
                chunk_size = 10000
                max_workers = min(os.cpu_count() or 4, 4)
                
                for chunk_start in range(0, total_pages, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, total_pages)
                    chunk_pages = chunk_end - chunk_start
                    
                    logging.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_pages + chunk_size - 1)//chunk_size}: pages {chunk_start + 1}-{chunk_end}")
                    
                    # Extract page data for this chunk only
                    page_data_list = []
                    output_paths = []
                    
                    for page_num in range(chunk_start, chunk_end):
                        try:
                            page = doc[page_num]
                            pix = page.get_pixmap(dpi=self.dpi, alpha=False)
                            page_data = pix.tobytes("png")
                            page_data_list.append(page_data)
                            
                            # Generate output path for this page
                            output_path = os.path.join(pdf_dir, f"{pdf_filename}_page_{page_num + 1:0{digits}d}.jpg")
                            output_paths.append(output_path)
                            
                            # Explicitly release pixmap and page
                            pix = None
                            page = None
                        except Exception as e:
                            logging.error(f"Failed to extract page {page_num+1}: {str(e)}")
                            page_data_list.append(None)
                            output_paths.append(None)
                    
                    # Process this chunk with ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        page_futures = {}
                        
                        # Submit pages in this chunk for processing
                        for i, (page_data, output_path) in enumerate(zip(page_data_list, output_paths)):
                            if page_data is not None and output_path is not None:
                                page_num = chunk_start + i
                                future = executor.submit(
                                    self._process_page_with_retries,
                                    page_data, 
                                    output_path, 
                                    page_num + 1, 
                                    total_pages,
                                    should_resize
                                )
                                page_futures[future] = (page_num, output_path)
                        
                        # Collect results for this chunk
                        successful_pages = 0
                        for future in as_completed(page_futures):
                            page_num, output_path = page_futures[future]
                            try:
                                page_success = future.result()
                                if page_success:
                                    converted_pages.append(output_path)
                                    successful_pages += 1
                                    logging.info(f"Saved page {page_num + 1} as {os.path.basename(output_path)}")
                                else:
                                    logging.error(f"Failed to process page {page_num + 1}")
                            except Exception as e:
                                logging.error(f"Error processing page {page_num + 1}: {str(e)}")
                    
                    # Clear chunk data from memory
                    page_data_list.clear()
                    page_data_list = None
                    output_paths = None
                    
                    # Force garbage collection after each chunk
                    for _ in range(3):
                        gc.collect()
                        time.sleep(0.1)
                    
                    logging.info(f"Completed chunk {chunk_start//chunk_size + 1}, processed {chunk_pages} pages")
                
                # Close document after all chunks are processed
                if doc:
                    try:
                        doc.close()
                        logging.info(f"PDF document closed")
                    except Exception as e:
                        logging.warning(f"Error closing PDF document: {str(e)}")
                    finally:
                        doc = None
                
                # Final cleanup
                for _ in range(5):
                    gc.collect()
                    time.sleep(0.5)
                
                # Additional Windows-specific cleanup
                if platform.system() == 'Windows':
                    try:
                        import ctypes
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                        time.sleep(2)
                    except Exception as e:
                        logging.warning(f"Error with Windows memory management: {str(e)}")
                
                # Check if all pages worked
                all_pages_converted = (len(converted_pages) == total_pages)
                if all_pages_converted:
                    logging.info(f"Successfully converted all {total_pages} pages from {pdf_path}")
                else:
                    logging.warning(f"Only converted {len(converted_pages)} out of {total_pages} pages from {pdf_path}")
                
                return converted_pages, all_pages_converted
            
            except Exception as e:
                logging.error(f"Error processing PDF {pdf_path}: {e}")
                logging.error(traceback.format_exc())
                return converted_pages, False
            
            finally:
                # cleanup pdf
                if 'doc' in locals() and doc:
                    try:
                        doc.close()
                        doc = None
                    except Exception as close_error:
                        logging.error(f"Error closing PDF {pdf_path}: {close_error}")
                
                # Final garbage collection
                for _ in range(3):
                    gc.collect()
                    time.sleep(0.5)
                
                # More aggressive memory cleanup on Windows
                if platform.system() == 'Windows':
                    try:
                        import ctypes
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                    except Exception as e:
                        logging.warning(f"Error freeing memory on Windows: {str(e)}")
                
                time.sleep(1)

    def process_folder(self, folder_path):
        """Process all PDFs in a folder with robust error handling"""
        # Create folder lock if it doesn't exist
        if folder_path not in self.folder_locks:
            self.folder_locks[folder_path] = threading.Lock()
            
        # Lock this folder for exclusive access
        with self.folder_locks[folder_path]:
            logging.info(f"Processing folder: {folder_path}")
            
            # Normalize the folder path to handle mixed slashes in network paths
            folder_path = os.path.normpath(folder_path)
            
            try:
                # Check if folder exists
                if not os.path.exists(folder_path):
                    logging.error(f"Folder does not exist: {folder_path}")
                    return (False, None)
                    
                # Verify folder is still stable (if it was marked stable before)
                if folder_path in self.stable_folders:
                    # Quick check to confirm nothing changed
                    try:
                        current_files = os.listdir(folder_path)
                        folder_check_passed = True
                    except Exception as e:
                        logging.error(f"Error checking folder contents: {str(e)}")
                        folder_check_passed = False
                        
                    if not folder_check_passed:
                        # Folder changed or disappeared, remove from stable set
                        self.stable_folders.discard(folder_path)
                        logging.info(f"Folder changed since stability check, will need to recheck")
                        
                # Check stability if not already verified
                if folder_path not in self.stable_folders:
                    if not check_folder_stability(folder_path):
                        logging.warning(f"Folder couldn't be verified as stable: {folder_path}")
                        return (False, None)
                    # Mark as stable once verified
                    self.stable_folders.add(folder_path)
                    logging.info(f"Folder verified as stable: {folder_path}")
                        
                # Get all files
                files = []
                try:
                    files = os.listdir(folder_path)
                    logging.info(f"Found {len(files)} files in folder")
                except Exception as e:
                    logging.error(f"Error listing folder contents: {str(e)}")
                    return (False, None)
                
                # Find PDFs - normalize paths for consistent processing
                pdf_files = [f for f in files if f.lower().endswith('.pdf')]
                
                # Add retry logic if no files were found
                if not pdf_files:
                    max_retries = 15
                    retry_interval = 2  # seconds
                    retry_count = 0
                    
                    logging.info(f"No PDF files found initially, will retry up to {max_retries} times")
                    
                    while retry_count < max_retries and not pdf_files:
                        retry_count += 1
                        logging.info(f"Retry attempt {retry_count}/{max_retries} for detecting PDFs in {folder_path}")
                        
                        # Wait before retrying
                        time.sleep(retry_interval)
                        
                        # Check again for files
                        if os.path.exists(folder_path):  # Make sure folder still exists
                            try:
                                files = os.listdir(folder_path)
                                pdf_files = [f for f in files if f.lower().endswith('.pdf')]
                                
                                if pdf_files:
                                    logging.info(f"Found {len(pdf_files)} PDFs on retry {retry_count}")
                                    break
                            except Exception as e:
                                logging.error(f"Error checking for PDFs during retry: {str(e)}")
                        else:
                            logging.warning(f"Folder disappeared during retry: {folder_path}")
                            return (False, None)
                
                # After all retries, check if we found any PDFs
                if not pdf_files:
                    logging.info(f"No PDF files found in {folder_path} after {retry_count} retries")
                    # No PDFs to delete, so safe to move folder
                    dest_folder = self._move_folder(folder_path)
                    return (True, dest_folder)
                
                # Process each PDF
                all_pdfs_deleted = True
                remaining_pdfs = []
                
                # Process each PDF
                for pdf_filename in pdf_files:
                    # Construct normalized path
                    pdf_path = os.path.normpath(os.path.join(folder_path, pdf_filename))
                    
                    # Verify file exists and is accessible
                    if not os.path.exists(pdf_path):
                        logging.warning(f"PDF file not found at normalized path: {pdf_path}")
                        # Try original path as a fallback
                        orig_path = os.path.join(folder_path, pdf_filename)
                        if os.path.exists(orig_path):
                            pdf_path = orig_path
                            logging.info(f"Found PDF at original path: {orig_path}")
                        else:
                            logging.error(f"PDF file not found at either path, skipping: {pdf_filename}")
                            continue
                    
                    logging.info(f"Processing PDF: {pdf_path}")
                    
                    try:
                        # Convert to JPEGs
                        converted_pages, all_pages_converted = self.process_pdf_to_jpgs(pdf_path)
                        
                        # Cleanup memory
                        gc.collect()
                        
                        # Wait for file handles to release - longer wait for network shares
                        time.sleep(5)
                        
                        # Remove original PDF if successful
                        if converted_pages and all_pages_converted:
                            # Try to delete original with enhanced removal
                            deletion_successful = safe_remove_file(pdf_path, max_retries=7, retry_delay=3)
                            
                            if not deletion_successful:
                                logging.warning(f"Could not delete PDF {pdf_path}")
                                all_pdfs_deleted = False
                                remaining_pdfs.append(pdf_path)
                        else:
                            logging.warning(f"Not deleting PDF {pdf_path} because conversion was incomplete")
                            all_pdfs_deleted = False
                            remaining_pdfs.append(pdf_path)
                    
                    except Exception as pdf_err:
                        logging.error(f"Error processing PDF {pdf_path}: {str(pdf_err)}")
                        logging.error(traceback.format_exc())
                        all_pdfs_deleted = False
                        remaining_pdfs.append(pdf_path)
                
                # Final check for PDFs in folder before moving
                final_check_pdfs = []
                try:
                    logging.info(f"Performing final check for any remaining PDFs in folder: {folder_path}")
                    # Get all current files in the folder
                    if os.path.exists(folder_path):
                        current_files = os.listdir(folder_path)
                        for filename in current_files:
                            file_path = os.path.join(folder_path, filename)
                            
                            # Check for .pdf extensions
                            if filename.lower().endswith('.pdf'):
                                final_check_pdfs.append(file_path)
                                logging.warning(f"PDF still present in final check: {file_path}")
                except Exception as check_err:
                    logging.error(f"Error in final check for PDFs: {str(check_err)}")
                
                # Only move the folder if no PDFs remain
                if all_pdfs_deleted and not final_check_pdfs:
                    logging.info(f"Verified no PDFs remain in folder. Moving to output directory.")
                    dest_folder = self._move_folder(folder_path)
                    
                    # Additional check after move to confirm everything worked
                    if dest_folder:
                        logging.info(f"Successfully moved folder to: {dest_folder}")
                        return (True, dest_folder)
                    else:
                        logging.error("Failed to move folder after processing")
                        return (False, None)
                else:
                    remaining = final_check_pdfs if final_check_pdfs else remaining_pdfs
                    logging.error(f"Not all PDFs were processed successfully. Remaining PDFs: {remaining}")
                    return (False, None)
                    
            except Exception as e:
                logging.error(f"Error processing folder {folder_path}: {str(e)}")
                logging.error(traceback.format_exc())
                return (False, None)
                
            finally:
                # Always clean up stable folders set if this folder is in it
                if folder_path in self.stable_folders:
                    self.stable_folders.discard(folder_path)
                    logging.info(f"Removed {folder_path} from stable folders set in cleanup")
                
                # Force garbage collection
                gc.collect()
                
    def _move_folder(self, folder_path):
        """Move processed folder to output location, stripping --noresize suffix"""
        try:
            folder_name = os.path.basename(folder_path)
            # Strip --noresize suffix (case insensitive)
            if folder_name.lower().endswith('--noresize'):
                clean_name = folder_name[:-10]  # Remove '--noresize'
            else:
                clean_name = folder_name
                
            dest_folder = os.path.join(self.output_directory, clean_name)
            
            # handle if dest already exists
            if os.path.exists(dest_folder):
                for item in os.listdir(folder_path):
                    src_item = os.path.join(folder_path, item)
                    dest_item = os.path.join(dest_folder, item)
                    
                    # rename if needed to avoid conflicts
                    if os.path.exists(dest_item):
                        base, ext = os.path.splitext(item)
                        counter = 1
                        while os.path.exists(dest_item):
                            new_filename = f"{base}_{counter}{ext}"
                            dest_item = os.path.join(dest_folder, new_filename)
                            counter += 1
                    
                    # Try to move with retries for network shares
                    move_success = False
                    for move_attempt in range(3):
                        try:
                            shutil.move(src_item, dest_item)
                            move_success = True
                            break
                        except Exception as move_err:
                            logging.warning(f"Move attempt {move_attempt+1} failed: {move_err}")
                            time.sleep(2)
                    
                    if not move_success:
                        logging.error(f"Failed to move {src_item} to {dest_item}")
                
                # remove source folder
                try:
                    # Wait before trying to remove the directory
                    time.sleep(2)
                    os.rmdir(folder_path)
                    logging.info(f"Merged contents of {folder_path} into {dest_folder}")
                except Exception as rmdir_error:
                    logging.error(f"Error removing source folder {folder_path}: {rmdir_error}")
            else:
                # just move the whole folder
                move_success = False
                for move_attempt in range(3):
                    try:
                        shutil.move(folder_path, dest_folder)
                        move_success = True
                        logging.info(f"Moved folder: {folder_path} -> {dest_folder}")
                        break
                    except Exception as move_err:
                        logging.warning(f"Folder move attempt {move_attempt+1} failed: {move_err}")
                        time.sleep(2)
                
                if not move_success:
                    logging.error(f"Failed to move folder {folder_path} to {dest_folder}")
                    return None
            
            # Return the destination folder path on success
            return dest_folder
            
        except Exception as e:
            logging.error(f"Error moving folder {folder_path}: {e}")
            return None
    
class FolderWatcher(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.processing_set = set()  # track folders being processed
        self.recently_completed = set()  # track folders that were recently completed
        self.completion_timestamps = {}  # track when folders were completed
        self.lock = threading.Lock()  # lock for thread-safe operations on shared sets

    def _cleanup_completed_folders(self):
        """Clean up the recently completed folders set after a time threshold"""
        current_time = time.time()
        to_remove = []
        
        with self.lock:
            for folder in self.recently_completed:
                timestamp = self.completion_timestamps.get(folder, 0)
                if current_time - timestamp > 60:  # Remove after 60 seconds
                    to_remove.append(folder)
            
            for folder in to_remove:
                self.recently_completed.remove(folder)
                if folder in self.completion_timestamps:
                    del self.completion_timestamps[folder]

    def on_created(self, event):
        if event.is_directory:
            folder_path = event.src_path
            
            # Clean up old entries in the completed set
            self._cleanup_completed_folders()
            
            # If this folder was recently processed and completed, ignore it
            with self.lock:
                if folder_path in self.recently_completed:
                    logging.info(f"Ignoring folder that was recently completed: {folder_path}")
                    return
            
            # Check if folder still exists before proceeding
            if not os.path.exists(folder_path):
                logging.warning(f"Folder no longer exists, skipping: {folder_path}")
                return
                
            # avoid processing same folder multiple times
            if folder_path in self.processing_set:
                logging.info(f"Folder already being processed: {folder_path}")
                return
                
            logging.info(f"New folder detected: {folder_path}")
            
            # add to processing set
            self.processing_set.add(folder_path)
            
            try:
                # run stability check if needed (will retry until stable)
                if folder_path not in self.processor.stable_folders:
                    if not check_folder_stability(folder_path):
                        logging.warning(f"Failed to verify folder stability: {folder_path}")
                        self.processing_set.discard(folder_path)
                        return
                    # mark as stable
                    self.processor.stable_folders.add(folder_path)
                    logging.info(f"Folder verified as stable and queued: {folder_path}")
                
                # process the folder - get the destination folder if moved successfully
                result = self.processor.process_folder(folder_path)
                
                # Handle the result
                if isinstance(result, tuple):
                    success, dest_folder = result
                else:
                    success = result
                    dest_folder = None
                
                # If processing was successful, add both original and destination paths to recently completed set
                if success:
                    with self.lock:
                        # Add the original folder path
                        self.recently_completed.add(folder_path)
                        self.completion_timestamps[folder_path] = time.time()
                        logging.info(f"Added folder to recently completed set: {folder_path}")
                        
                        # If we know where it was moved to, add that path as well
                        if dest_folder:
                            self.recently_completed.add(dest_folder)
                            self.completion_timestamps[dest_folder] = time.time()
                            logging.info(f"Added destination folder to recently completed set: {dest_folder}")
            except Exception as e:
                logging.error(f"Error processing folder {folder_path}: {e}")
                logging.error(traceback.format_exc())
            finally:
                # always clean up processing set
                self.processing_set.discard(folder_path)

    def on_modified(self, event):
        if event.is_directory:
            folder_path = event.src_path
            
            # Clean up old entries in the completed set
            self._cleanup_completed_folders()
            
            # If this folder was recently processed and completed, ignore it
            with self.lock:
                if folder_path in self.recently_completed:
                    logging.info(f"Ignoring modified folder that was recently completed: {folder_path}")
                    return
            
            # Check if folder still exists before proceeding
            if not os.path.exists(folder_path):
                logging.warning(f"Modified folder no longer exists, skipping: {folder_path}")
                return
                
            # avoid processing same folder multiple times
            if folder_path in self.processing_set:
                logging.info(f"Folder already being processed: {folder_path}")
                return
                
            # check if it contains files we process
            try:
                if not os.path.exists(folder_path):
                    return
                    
                pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                
                if not pdf_files:
                    return
            except Exception as e:
                logging.error(f"Error checking folder contents: {str(e)}")
                return
                
            logging.info(f"Folder modified with PDFs: {folder_path}")
            
            # add to processing set
            self.processing_set.add(folder_path)
            
            try:
                # remove from stable folders if it was previously marked stable
                self.processor.stable_folders.discard(folder_path)
                
                # check stability (will retry until stable)
                if not check_folder_stability(folder_path):
                    logging.warning(f"Failed to verify folder stability after modification: {folder_path}")
                    self.processing_set.discard(folder_path)
                    return
                
                # mark as stable
                self.processor.stable_folders.add(folder_path)
                
                # process the folder - get the destination folder if moved successfully
                result = self.processor.process_folder(folder_path)
                
                # Handle the result
                if isinstance(result, tuple):
                    success, dest_folder = result
                else:
                    success = result
                    dest_folder = None
                
                # If processing was successful, add both original and destination paths to recently completed set
                if success:
                    with self.lock:
                        # Add the original folder path
                        self.recently_completed.add(folder_path)
                        self.completion_timestamps[folder_path] = time.time()
                        
                        # If we know where it was moved to, add that path as well
                        if dest_folder:
                            self.recently_completed.add(dest_folder)
                            self.completion_timestamps[dest_folder] = time.time()
                            logging.info(f"Added both original and destination folders to recently completed set")
            except Exception as e:
                logging.error(f"Error processing modified folder {folder_path}: {e}")
                logging.error(traceback.format_exc())
            finally:
                # always clean up processing set
                self.processing_set.discard(folder_path)

def main():
    # get args
    args = parse_args()
    
    # check directories
    if not os.path.exists(args.watch_dir):
        logging.error(f"Watch directory does not exist: {args.watch_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")
    
    # create processor
    processor = PDFProcessor(args.watch_dir, args.output_dir, dpi=args.dpi if hasattr(args, 'dpi') else 200)
    
    # process any existing folders in parallel if max_workers > 1
    existing_folders = [os.path.join(args.watch_dir, item) for item in os.listdir(args.watch_dir) 
                       if os.path.isdir(os.path.join(args.watch_dir, item))]
    
    if args.max_workers > 1 and existing_folders:
        logging.info(f"Processing {len(existing_folders)} existing folders with {args.max_workers} workers")
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(processor.process_folder, folder) for folder in existing_folders]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing existing folder: {str(e)}")
    else:
        # Process sequentially
        for item_path in existing_folders:
            processor.process_folder(item_path)
    
    # setup file watcher
    event_handler = FolderWatcher(processor)
    observer = Observer()
    observer.schedule(event_handler, args.watch_dir, recursive=False)
    observer.start()
    
    logging.info(f"Watching directory: {args.watch_dir}")
    
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == "__main__":
    main()