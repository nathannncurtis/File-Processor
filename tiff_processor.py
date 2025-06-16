import os
import sys
import time
import shutil
import threading
import platform
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fitz  # PyMuPDF
from PIL import Image, ImageFile
import numpy as np 
import cv2
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc # for garbage collection
import traceback

# make PIL handle broken images better
ImageFile.LOAD_TRUNCATED_IMAGES = True

# setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tiff_processor.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
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
        
def needs_resizing(image_path, target_width=1700, target_height=2200):
    """Check if image already meets target dimensions"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            # Check if already correct size (either orientation)
            if ((width == target_width and height == target_height) or 
                (width == target_height and height == target_width)):
                return False
                
        return True
    except Exception:
        return True  # If we can't check, assume it needs resizing

def resize_image_inplace(file_path, target_width=1700, target_height=2200, target_dpi=200):
    """Resize image in place with retries"""
    for attempt in range(3):
        try:
            with Image.open(file_path) as img:
                original_width, original_height = img.size
                original_is_landscape = original_width > original_height
                
                # Adjust target based on orientation
                if original_is_landscape and target_width < target_height:
                    target_width, target_height = target_height, target_width
                elif not original_is_landscape and target_width > target_height:
                    target_width, target_height = target_height, target_width

                # Calculate scaling
                scale = min(target_width / original_width, target_height / original_height)
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)

                # Resize and center
                scaled_img = img.resize((new_width, new_height), Image.LANCZOS)
                final_img = Image.new('RGB', (target_width, target_height), 'white')
                
                x_offset = (target_width - new_width) // 2
                y_offset = (target_height - new_height) // 2
                final_img.paste(scaled_img, (x_offset, y_offset))

                # Convert to grayscale for TIFF processing
                final_img = final_img.convert('L')

                # Save as temporary file first (will be processed to TIFF later)
                final_img.save(file_path, "JPEG", quality=95, dpi=(target_dpi, target_dpi))
                
                del scaled_img, final_img
                gc.collect()
                return True
                
        except Exception as e:
            if attempt < 2:
                time.sleep(1)  # 1 second delay before retry
                continue
            else:
                logging.warning(f"Failed to resize {file_path} after 3 attempts: {e}")
                return False
    return False

def parse_args():
    parser = argparse.ArgumentParser(description="TIFF Processor for PDFs and JPEGs")
    parser.add_argument('--watch-dir', required=True, help='Directory to watch for PDF/JPEG folders')
    parser.add_argument('--output-dir', required=True, help='Output directory for processed folders')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum number of worker threads for processing (default: 1)')
    parser.add_argument('--dpi', type=int, default=200, help='DPI for TIFF conversion (default: 200)')
    return parser.parse_args()

class PDFProcessor:
    def __init__(self, watch_directory, output_directory, dpi=200):
        self.watch_directory = watch_directory
        self.output_directory = output_directory
        self.dpi = dpi
        self.success_count = 0
        self.failure_count = 0
        self.processed_folders = set()  # track folders we've already processed
        self.stable_folders = set()  # track folders that have been verified as stable
        self.lock = threading.Lock()  # lock for thread-safe operations
        self.file_locks = {}  # for per-PDF locks
        self.folder_locks = {}  # for per-folder locks

    def _validate_tiff(self, file_path):
        """check if tiff file is valid and can be opened"""
        try:
            # first check basics - file exists and has size
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logging.error(f"TIFF file {file_path} is empty or doesn't exist")
                return False
                
            # try opening the file to make sure it's usable
            with Image.open(file_path) as img:
                img.verify()  # verify instead of load to catch corrupted data
                # make sure it's actually a tiff
                if img.format != "TIFF":
                    logging.error(f"File {file_path} is not a valid TIFF format")
                    return False
            return True
        except Exception as e:
            logging.error(f"Invalid TIFF file {file_path}: {str(e)}")
            return False

    def _safe_remove(self, file_path, max_retries=3, retry_delay=1):
        """try to remove a file with retries in case it's locked"""
        if not os.path.exists(file_path):
            return
            
        for attempt in range(max_retries):
            try:
                os.remove(file_path)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.debug(f"Retry {attempt+1} removing file {file_path}: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    logging.warning(f"Failed to remove file {file_path}: {str(e)}")

    def _convert_page(self, page, output_path, should_resize=True):
        """convert one pdf page to tiff with opencv adaptive thresholding and group4 compression"""
        temp_png = None
        temp_jpeg = None
        try:
            # create temp file for intermediate processing
            temp_dir = os.path.dirname(output_path)
            temp_basename = f"temp_{os.path.basename(output_path).replace('.tif', '')}_{int(time.time()*1000)}.png"
            temp_png = os.path.join(temp_dir, temp_basename)
            
            # generate pixmap with our dpi settings
            pix = page.get_pixmap(dpi=self.dpi, alpha=False)
            
            # save as png using memory buffer to avoid file I/O when possible
            try:
                # Try the memory buffer approach first
                img_data = pix.tobytes("png")
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                
                # Explicitly release pixmap memory
                pix = None
                
                # Ensure we got a valid image
                if img is None:
                    raise ValueError("Failed to decode image from memory buffer")
            except Exception:
                # Fall back to file-based approach if memory buffer fails
                pix.pil_save(temp_png, format="PNG")
                
                # Explicitly release pixmap memory
                pix = None
                
                # Force garbage collection after releasing pixmap
                gc.collect()
                
                # Wait briefly to ensure file is fully written
                time.sleep(0.1)
                
                # Check temp file was created properly
                if not os.path.exists(temp_png) or os.path.getsize(temp_png) == 0:
                    raise ValueError(f"Failed to create temporary PNG file: {temp_png}")
                
                # Read with OpenCV
                img = cv2.imread(temp_png, cv2.IMREAD_GRAYSCALE)
                
            if img is None:
                raise ValueError(f"Failed to load image: Memory buffer and file approaches both failed")
            
            # Resize before thresholding if needed
            if should_resize:
                # Save as temporary JPEG for resizing
                temp_jpeg_basename = f"temp_resize_{os.path.basename(output_path).replace('.tif', '')}_{int(time.time()*1000)}.jpg"
                temp_jpeg = os.path.join(temp_dir, temp_jpeg_basename)
                
                # Convert opencv image to PIL and save
                pil_img = Image.fromarray(img)
                pil_img.save(temp_jpeg, "JPEG", quality=95, dpi=(self.dpi, self.dpi))
                del pil_img
                
                # Check if resizing is needed and do it
                if needs_resizing(temp_jpeg):
                    resize_image_inplace(temp_jpeg)
                
                # Read back the resized image
                img = cv2.imread(temp_jpeg, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Failed to read resized image")
                
            # Check if the page is predominantly dark
            # Calculate average brightness of the image
            avg_brightness = np.mean(img)
            is_dark_page = avg_brightness < 50  # threshold for considering a page "dark"
            
            if is_dark_page:
                logging.info(f"Detected predominantly dark page, using special processing")
                # For dark pages, use global thresholding instead of adaptive
                # This preserves the black background while still capturing any light content
                _, binary = cv2.threshold(
                    img,
                    40,  # threshold value, pixels below this become black (0)
                    255,  # max value for pixels above threshold
                    cv2.THRESH_BINARY  # binary output
                )
            else:
                # For normal pages, use adaptive thresholding as before
                binary = cv2.adaptiveThreshold(
                    img, 
                    255,  # max value
                    cv2.ADAPTIVE_THRESH_MEAN_C,  # mean thresholding
                    cv2.THRESH_BINARY,  # binary output
                    15,  # block size (keep same as before, must be odd)
                    5    # c value - constant subtracted from mean
                )
            
            # Free memory for the original image
            img = None
            
            # convert to pil for saving with group 4 compression
            binary_img = Image.fromarray(binary).convert('1')
            
            # Free memory for the binary array
            binary = None
            
            # save with group4 compression
            binary_img.save(
                output_path, 
                "TIFF", 
                compression="group4", 
                dpi=(self.dpi, self.dpi)
            )
            
            # Free memory for the PIL image
            binary_img = None
            
            # Force garbage collection after processing
            gc.collect()
            
            # verify the tiff is good
            if not self._validate_tiff(output_path):
                raise ValueError(f"TIFF validation failed for {output_path}")
                
            return True
        except Exception as e:
            logging.error(f"PDF page conversion failed: {str(e)}")
            # clean up any partial output
            if os.path.exists(output_path):
                self._safe_remove(output_path)
            return False
        finally:
            # always clean up temp files
            if temp_png and os.path.exists(temp_png):
                self._safe_remove(temp_png)
            if temp_jpeg and os.path.exists(temp_jpeg):
                self._safe_remove(temp_jpeg)

    def _convert_jpeg_to_tiff(self, jpeg_path, output_path, should_resize=True):
        """convert a jpeg file to tiff using opencv adaptive thresholding with group4 compression"""
        temp_jpeg = None
        try:
            # Normalize paths
            jpeg_path = os.path.normpath(jpeg_path)
            output_path = os.path.normpath(output_path)
            
            # Check if file exists
            if not os.path.exists(jpeg_path):
                logging.error(f"JPEG file does not exist: {jpeg_path}")
                return False
                
            # Check if output already exists in a valid form
            if os.path.exists(output_path):
                if self._validate_tiff(output_path):
                    logging.info(f"Valid TIFF already exists at {output_path}, skipping conversion")
                    return True
                else:
                    # Remove invalid output file
                    logging.warning(f"Removing invalid existing TIFF: {output_path}")
                    self._safe_remove(output_path)
            
            # Handle resizing if needed
            working_jpeg_path = jpeg_path
            if should_resize and needs_resizing(jpeg_path):
                # Create temp resized copy
                temp_dir = os.path.dirname(jpeg_path)
                temp_basename = f"temp_resize_{os.path.basename(jpeg_path)}_{int(time.time()*1000)}.jpg"
                temp_jpeg = os.path.join(temp_dir, temp_basename)
                
                # Copy original to temp for resizing
                shutil.copy2(jpeg_path, temp_jpeg)
                
                # Resize the temp copy
                if resize_image_inplace(temp_jpeg):
                    working_jpeg_path = temp_jpeg
                else:
                    # If resize failed, use original
                    working_jpeg_path = jpeg_path
            
            # Load image for thresholding
            img = cv2.imread(working_jpeg_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image with OpenCV: {working_jpeg_path}")
            
            # Check if the image is predominantly dark
            avg_brightness = np.mean(img)
            is_dark_image = avg_brightness < 50  # threshold for considering an image "dark"
            
            if is_dark_image:
                logging.info(f"Detected predominantly dark JPEG, using special processing")
                # For dark images, use global thresholding instead of adaptive
                _, binary = cv2.threshold(
                    img,
                    40,  # threshold value
                    255,  # max value
                    cv2.THRESH_BINARY  # binary output
                )
            else:
                # For normal images, use adaptive thresholding
                binary = cv2.adaptiveThreshold(
                    img,
                    255,  # max value
                    cv2.ADAPTIVE_THRESH_MEAN_C,  # mean thresholding
                    cv2.THRESH_BINARY,  # binary output
                    15,  # block size (must be odd)
                    5    # c value - constant subtracted from mean
                )
            
            # Free memory for original image
            img = None
            
            # Convert to pil for saving with group 4
            binary_img = Image.fromarray(binary).convert('1')
            
            # Free memory for binary array
            binary = None
            
            # Create temp output path to avoid partial writes
            temp_output = output_path + ".temp"
            
            # Save with group4 compression to temp file first
            binary_img.save(
                temp_output, 
                "TIFF", 
                compression="group4", 
                dpi=(self.dpi, self.dpi)
            )
            
            # Free memory for PIL image
            binary_img = None
            
            # Force garbage collection
            gc.collect()
            
            # Verify temp file is valid before moving
            if self._validate_tiff(temp_output):
                # Move temp file to final destination
                try:
                    os.replace(temp_output, output_path)
                    logging.info(f"Moved validated TIFF to final destination: {output_path}")
                except Exception as move_err:
                    logging.error(f"Error moving temp TIFF to final location: {str(move_err)}")
                    if os.path.exists(temp_output):
                        self._safe_remove(temp_output)
                    return False
            else:
                logging.error(f"Temp TIFF validation failed: {temp_output}")
                if os.path.exists(temp_output):
                    self._safe_remove(temp_output)
                return False
                
            # Final validation check on destination file
            if not self._validate_tiff(output_path):
                raise ValueError("TIFF validation failed after move")
                
            return True
        except Exception as e:
            logging.error(f"JPEG to TIFF conversion failed for {jpeg_path}: {str(e)}")
            if os.path.exists(output_path):
                self._safe_remove(output_path)
            return False
        finally:
            # Clean up temp resized JPEG if created
            if temp_jpeg and os.path.exists(temp_jpeg):
                self._safe_remove(temp_jpeg)
        
    def process_jpeg(self, jpeg_path):
        """process a single jpeg file to tiff - must succeed"""
        # Normalize the path for consistent locking
        jpeg_path = os.path.normpath(jpeg_path)
        
        # Create file lock if it doesn't exist
        if jpeg_path not in self.file_locks:
            self.file_locks[jpeg_path] = threading.Lock()
            
        # Lock this specific JPEG file for exclusive access
        with self.file_locks[jpeg_path]:
            try:
                logging.info(f"Processing jpeg: {jpeg_path}")
                
                # Check if folder wants resizing
                folder_name = os.path.basename(os.path.dirname(jpeg_path))
                should_resize = not folder_name.lower().endswith('--noresize')
                
                # Check if output already exists (from a previous run)
                output_path = jpeg_path.rsplit('.', 1)[0] + '.tif'
                if os.path.exists(output_path) and self._validate_tiff(output_path):
                    logging.info(f"TIFF already exists and is valid, skipping conversion: {output_path}")
                    # Still count this as a success
                    with self.lock:
                        self.success_count += 1
                    # No need to delete original if we didn't do the conversion
                    return True
                
                # Infinite retry loop - never give up
                attempt = 0
                success = False
                
                while not success:
                    attempt += 1
                    if attempt > 1:
                        logging.warning(f"Retry attempt {attempt} for jpeg: {jpeg_path}")
                        time.sleep(1)  # pause before retry
                        
                    try:
                        success = self._convert_jpeg_to_tiff(jpeg_path, output_path, should_resize)
                        
                        # Verify the file exists and is valid
                        if success and os.path.exists(output_path) and self._validate_tiff(output_path):
                            # Conversion worked, break out of retry loop
                            break
                    except Exception as e:
                        logging.error(f"Attempt {attempt} failed: {str(e)}")
                        time.sleep(1)  # Sleep before retry
                        
                    # Add a safety limit to prevent infinite loops
                    if attempt >= 5:
                        logging.error(f"Failed after {attempt} attempts, giving up on: {jpeg_path}")
                        return False
                
                # Conversion succeeded
                with self.lock:
                    self.success_count += 1
                logging.info(f"Successfully converted jpeg: {jpeg_path}")
                self._safe_remove(jpeg_path)
                return True
                        
            except Exception as e:
                with self.lock:
                    self.failure_count += 1
                logging.error(f"CRITICAL ERROR: jpeg processing failed for {jpeg_path}: {str(e)}")
                return False

    def _convert_page_with_retries(self, page, output_path, page_num, total_pages, should_resize=True):
        """Process a single page with infinite retries"""
        attempt = 0
        
        # infinite retry loop - never give up on a page
        while True:
            attempt += 1
            if attempt > 1:
                logging.warning(f"retry attempt {attempt} for page {page_num} of {total_pages}")
                time.sleep(1)  # brief pause before retry
            
            try:
                # process the page
                page_success = self._convert_page(page, output_path, should_resize)
                
                if page_success and os.path.exists(output_path) and self._validate_tiff(output_path):
                    # page converted successfully
                    logging.info(f"successfully converted page {page_num} of {total_pages}")
                    return True
                else:
                    logging.warning(f"failed attempt {attempt} for page {page_num}")
            except Exception as e:
                logging.error(f"error on attempt {attempt} for page {page_num}: {str(e)}")
        
        # This line should never be reached because the loop only exits with a return
        return False

    def process_pdf(self, pdf_path):
        """convert pdf file to tiff files - must convert ALL pages"""
        # Create file lock if it doesn't exist
        if pdf_path not in self.file_locks:
            self.file_locks[pdf_path] = threading.Lock()
            
        # Lock this PDF for exclusive access
        with self.file_locks[pdf_path]:
            success = True
            created_files = []
            doc = None
            
            # Check if folder wants resizing
            folder_name = os.path.basename(os.path.dirname(pdf_path))
            should_resize = not folder_name.lower().endswith('--noresize')
            
            try:
                # Check file size to determine if memory mapping would be beneficial
                # (memory parameter will be used differently depending on PyMuPDF version)
                file_size = os.path.getsize(pdf_path)
                
                # open the pdf file - try to use memory mapping if available in this PyMuPDF version
                try:
                    # Try with the memory parameter (newer versions of PyMuPDF)
                    doc = fitz.open(pdf_path, filetype="pdf", memory=file_size > 50_000_000)
                except TypeError:
                    # Fallback for older versions that don't support the memory parameter
                    doc = fitz.open(pdf_path)
                total_pages = len(doc)
                digits = len(str(total_pages)) + 1
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_dir = os.path.dirname(pdf_path)

                logging.info(f"processing pdf: {pdf_path} ({total_pages} pages)")
                
                # Pre-generate output paths for all pages
                output_paths = [os.path.join(output_dir, f"{base_name}_page_{page_num+1:0{digits}d}.tif") 
                            for page_num in range(total_pages)]
                
                # Process pages in parallel but track results by page number
                # Choose appropriate number of workers based on CPU cores, but avoid too many
                max_workers = min(os.cpu_count() or 4, 4)  # Limit to 4 threads max
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Map to store futures by page number
                    page_futures = {}
                    
                    # Submit all pages for processing
                    for page_num in range(total_pages):
                        future = executor.submit(
                            self._convert_page_with_retries, 
                            doc[page_num], 
                            output_paths[page_num], 
                            page_num + 1, 
                            total_pages,
                            should_resize  # Add resize parameter
                        )
                        page_futures[future] = (page_num, output_paths[page_num])
                    
                    # Collect results as they complete (order doesn't matter here)
                    for future in as_completed(page_futures):
                        page_num, output_path = page_futures[future]
                        try:
                            page_success = future.result()
                            if page_success:
                                created_files.append(output_path)
                                with self.lock:
                                    self.success_count += 1
                            else:
                                # This should never happen due to infinite retries, but handle it anyway
                                logging.error(f"Failed to process page {page_num+1} despite retries")
                                success = False
                        except Exception as e:
                            logging.error(f"Error processing page {page_num+1}: {str(e)}")
                            success = False
                
                # close the document to release file handles
                if doc:
                    doc.close()
                    doc = None  # Explicitly set to None
                
                # force garbage collection to help release handles
                gc.collect()
                
                # More aggressive memory cleanup on Windows
                if platform.system() == 'Windows':
                    try:
                        import ctypes
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                    except Exception as e:
                        logging.warning(f"Error freeing memory on Windows: {str(e)}")
                
                # final status check - all pages must be converted
                if len(created_files) == total_pages:
                    logging.info(f"all {total_pages} pages of {pdf_path} processed successfully")
                    
                    # Brief pause to ensure file handles are fully released
                    time.sleep(2)
                    
                    # Try delete with additional retries
                    delete_success = self._safe_remove_with_retries(pdf_path, max_retries=5, retry_delay=2)
                    if not delete_success:
                        logging.warning(f"Could not delete PDF after multiple attempts: {pdf_path}")
                        
                    return True
                else:
                    # some pages failed, don't delete the PDF but also don't consider this a success
                    logging.error(f"CRITICAL ERROR: Not all pages converted for {pdf_path}, original preserved")
                    return False
                    
            except Exception as e:
                success = False
                with self.lock:
                    self.failure_count += 1
                logging.error(f"CRITICAL ERROR: pdf processing failed for {pdf_path}: {str(e)}")
                return False
                    
            finally:
                # always close the pdf to avoid memory leaks
                if doc:
                    try:
                        doc.close()
                    except Exception as e:
                        logging.warning(f"error closing pdf document: {str(e)}")
                
                # Force garbage collection after full PDF processing
                gc.collect()

    def _safe_remove_with_retries(self, file_path, max_retries=5, retry_delay=2):
        """try harder to remove a file with more retries and delay"""
        if not os.path.exists(file_path):
            return True
            
        for attempt in range(max_retries):
            try:
                os.remove(file_path)
                logging.info(f"Successfully removed file after attempt {attempt+1}: {file_path}")
                return True
            except Exception as e:
                logging.warning(f"Attempt {attempt+1} to remove file failed: {file_path}, Error: {str(e)}")
                
                # Force garbage collection on each attempt
                gc.collect()
                
                if attempt < max_retries - 1:
                    logging.info(f"Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
        
        logging.error(f"Failed to remove file after {max_retries} attempts: {file_path}")
        return False

    def process_folder(self, folder_path):
        """process all pdfs and jpegs in a folder"""
        # Create folder lock if it doesn't exist
        if folder_path not in self.folder_locks:
            self.folder_locks[folder_path] = threading.Lock()
            
        # Lock this folder for exclusive access
        with self.folder_locks[folder_path]:
            try:
                # check folder still exists
                if not os.path.exists(folder_path):
                    logging.info(f"Folder no longer exists: {folder_path}")
                    return False
                    
                # verify folder is still stable (if it was marked stable before)
                if folder_path in self.stable_folders:
                    # quick check to confirm nothing changed
                    try:
                        current_files = os.listdir(folder_path)
                        folder_check_passed = True
                    except Exception:
                        folder_check_passed = False
                        
                    if not folder_check_passed:
                        # folder changed or disappeared, remove from stable set
                        self.stable_folders.discard(folder_path)
                        logging.info(f"Folder changed since stability check, will need to recheck")
                        
                # check stability if not already verified
                if folder_path not in self.stable_folders:
                    if not check_folder_stability(folder_path):
                        logging.warning(f"Folder couldn't be verified as stable: {folder_path}")
                        return False
                    # mark as stable once verified
                    self.stable_folders.add(folder_path)
                    logging.info(f"Folder verified as stable: {folder_path}")
                    
                # see if there's anything to process
                pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
                
                # Add retry logic if no files were found
                if not pdf_files and not jpeg_files:
                    max_retries = 15
                    retry_interval = 2  # seconds
                    retry_count = 0
                    
                    logging.info(f"No PDF or JPEG files found initially, will retry up to {max_retries} times")
                    
                    while retry_count < max_retries and not (pdf_files or jpeg_files):
                        retry_count += 1
                        logging.info(f"Retry attempt {retry_count}/{max_retries} for detecting files in {folder_path}")
                        
                        # Wait before retrying
                        time.sleep(retry_interval)
                        
                        # Check again for files
                        if os.path.exists(folder_path):  # Make sure folder still exists
                            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                            jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
                            
                            if pdf_files or jpeg_files:
                                logging.info(f"Found processable files on retry {retry_count}: {len(pdf_files)} PDFs, {len(jpeg_files)} JPEGs")
                                break
                        else:
                            logging.warning(f"Folder disappeared during retry: {folder_path}")
                            return False
                    
                    # After all retries, if there's still nothing to process
                    if not pdf_files and not jpeg_files:
                        # nothing to process - safe to move folder
                        if folder_path in self.processed_folders:
                            logging.info(f"Folder already processed and has no new files: {folder_path}")
                            return True
                        else:
                            logging.info(f"No PDF or JPEG files found in folder after {retry_count} retries: {folder_path}")
                            self._move_folder(folder_path)
                            self.processed_folders.add(folder_path)
                            return True
                
                # got files to process
                if folder_path in self.processed_folders:
                    logging.info(f"Folder previously processed but contains new files: {folder_path}")
                    self.processed_folders.remove(folder_path)
                
                # Get updated list of files (in case they changed during stability check)
                pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                pdf_results = []
                pdf_deletion_success = True
                remaining_pdfs = []
                
                # Process PDFs one by one (each PDF is already processed in parallel internally)
                for pdf_file in pdf_files:
                    pdf_path = os.path.join(folder_path, pdf_file)
                    result = self.process_pdf(pdf_path)
                    pdf_results.append(result)
                    
                    # Check if the PDF file still exists after processing
                    if os.path.exists(pdf_path) or os.path.exists(pdf_path + ".processed"):
                        pdf_deletion_success = False
                        if os.path.exists(pdf_path):
                            remaining_pdfs.append(pdf_path)
                        else:
                            remaining_pdfs.append(pdf_path + ".processed")
                    
                    # Force garbage collection after each PDF
                    gc.collect()

                # Process JPEGs in parallel
                jpeg_files = [f for f in os.listdir(folder_path) 
                            if f.lower().endswith(('.jpg', '.jpeg'))]
                jpeg_results = []
                
                # For small numbers of JPEGs, just process them sequentially
                # For larger batches, process them in parallel
                if len(jpeg_files) <= 3:
                    for jpeg_file in jpeg_files:
                        jpeg_path = os.path.join(folder_path, jpeg_file)
                        result = self.process_jpeg(jpeg_path)
                        jpeg_results.append(result)
                else:
                    # Use parallel processing for larger batches
                    max_workers = min(os.cpu_count() or 4, 4)  # Limit to 4 threads max
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        jpeg_paths = [os.path.join(folder_path, jpeg_file) for jpeg_file in jpeg_files]
                        futures = [executor.submit(self.process_jpeg, jpeg_path) for jpeg_path in jpeg_paths]
                        
                        for future in futures:
                            try:
                                result = future.result()
                                jpeg_results.append(result)
                            except Exception as e:
                                logging.error(f"Error in JPEG processing: {str(e)}")
                                jpeg_results.append(False)

                # Check if all processing was successful and PDFs were deleted
                all_successful = True
                if pdf_files or jpeg_files:
                    if not all(pdf_results + jpeg_results):
                        all_successful = False
                        logging.warning(f"Not all files in {folder_path} were processed successfully")
                    
                    # Only move the folder if successful and all PDFs were deleted
                    if all_successful and pdf_deletion_success:
                        logging.info(f"All files processed successfully and all PDFs deleted. Moving folder.")
                        self._move_folder(folder_path)
                        self.processed_folders.add(folder_path)  # mark as done
                    elif not pdf_deletion_success:
                        logging.warning(f"Not all PDFs were deleted. Remaining PDFs: {remaining_pdfs}")
                        return False
                else:
                    # empty folders just get moved
                    logging.info(f"No PDF or JPEG files found in {folder_path}")
                    self._move_folder(folder_path)
                    self.processed_folders.add(folder_path)
                    
                # remove from stable folders after processing
                self.stable_folders.discard(folder_path)
                
                # Force garbage collection after processing folder
                gc.collect()
                
                # More aggressive memory cleanup on Windows
                if platform.system() == 'Windows':
                    try:
                        import ctypes
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                    except Exception as e:
                        logging.warning(f"Error freeing memory on Windows: {str(e)}")
                    
                return all_successful and pdf_deletion_success

            except Exception as e:
                logging.error(f"Folder processing failed: {str(e)}")
                return False

    def _move_folder(self, src_folder):
        """move processed folder to output location, merging if destination already exists"""
        try:
            folder_name = os.path.basename(src_folder)
            # Strip --noresize suffix (case insensitive)
            if folder_name.lower().endswith('--noresize'):
                clean_name = folder_name[:-10]  # Remove '--noresize'
            else:
                clean_name = folder_name
                
            dest_folder = os.path.join(self.output_directory, clean_name)
            
            if os.path.exists(dest_folder):
                logging.info(f"destination exists, merging contents: {dest_folder}")
                # merge the folders - move all items from src to dest
                for item in os.listdir(src_folder):
                    src_item = os.path.join(src_folder, item)
                    dest_item = os.path.join(dest_folder, item)
                    
                    # handle file conflict by renaming if needed
                    if os.path.exists(dest_item):
                        # rename by adding counter suffix
                        base, ext = os.path.splitext(item)
                        counter = 1
                        new_dest_item = dest_item
                        while os.path.exists(new_dest_item):
                            new_name = f"{base}_{counter}{ext}"
                            new_dest_item = os.path.join(dest_folder, new_name)
                            counter += 1
                        
                        # move with new name
                        shutil.move(src_item, new_dest_item)
                        logging.info(f"renamed and moved: {src_item} -> {new_dest_item}")
                    else:
                        # move directly
                        shutil.move(src_item, dest_item)
                        logging.info(f"moved: {src_item} -> {dest_item}")
                
                # remove source folder after moving everything
                try:
                    os.rmdir(src_folder)
                    logging.info(f"removed empty source folder: {src_folder}")
                except Exception as rm_error:
                    logging.error(f"error removing source folder: {str(rm_error)}")
            else:
                # if destination doesn't exist, just move the entire folder
                # make sure parent directory exists
                os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
                
                # simple move
                shutil.move(src_folder, dest_folder)
                logging.info(f"moved folder: {src_folder} -> {dest_folder}")
                
            return True
        except Exception as e:
            logging.error(f"folder move failed: {str(e)}")
            return False

    def _merge_folders(self, src_folder, dest_folder):
        """merge source folder into destination folder, handling file conflicts"""
        try:
            # make sure destination exists
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
                
            # move all files from source to destination
            for item in os.listdir(src_folder):
                src_item = os.path.join(src_folder, item)
                dest_item = os.path.join(dest_folder, item)
                
                if os.path.isdir(src_item):
                    # recursively merge subdirectories
                    self._merge_folders(src_item, dest_item)
                else:
                    # handle file conflict by renaming if needed
                    if os.path.exists(dest_item):
                        # rename by adding counter suffix
                        base, ext = os.path.splitext(item)
                        counter = 1
                        new_dest_item = dest_item
                        while os.path.exists(new_dest_item):
                            new_name = f"{base}_{counter}{ext}"
                            new_dest_item = os.path.join(dest_folder, new_name)
                            counter += 1
                        
                        # move with new name
                        shutil.move(src_item, new_dest_item)
                        logging.info(f"renamed and moved file: {src_item} -> {new_dest_item}")
                    else:
                        # move file directly
                        shutil.move(src_item, dest_item)
                        logging.info(f"moved file: {src_item} -> {dest_item}")
            
            # remove source folder after all items have been moved
            if os.path.exists(src_folder) and len(os.listdir(src_folder)) == 0:
                os.rmdir(src_folder)
                logging.info(f"removed empty source folder: {src_folder}")
            
            return True
        except Exception as e:
            logging.error(f"folder merge failed: {str(e)}")
            return False

class FolderHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.processing_set = set()  # track folders being processed

    def on_created(self, event):
        if event.is_directory:
            folder_path = event.src_path
            
            # avoid processing same folder multiple times
            if folder_path in self.processing_set:
                logging.info(f"folder already being processed: {folder_path}")
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
                
                # reset if folder gets recreated
                if folder_path in self.processor.processed_folders:
                    self.processor.processed_folders.remove(folder_path)
                
                # process the folder
                self.processor.process_folder(folder_path)
            except Exception as e:
                logging.error(f"error processing folder {folder_path}: {e}")
                logging.error(traceback.format_exc())
            finally:
                # always clean up processing set
                self.processing_set.discard(folder_path)

    def on_modified(self, event):
        if event.is_directory:
            folder_path = event.src_path
            
            # avoid processing same folder multiple times
            if folder_path in self.processing_set:
                logging.info(f"folder already being processed: {folder_path}")
                return
                
            # check if it contains files we process
            try:
                if not os.path.exists(folder_path):
                    return
                    
                pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
                jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
                
                if not pdf_files and not jpeg_files:
                    return
            except Exception as e:
                logging.error(f"Error checking folder contents: {str(e)}")
                return
                
            logging.info(f"Folder modified with processable files: {folder_path}")
            
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
                
                # reset if folder is modified with new files
                if folder_path in self.processor.processed_folders:
                    self.processor.processed_folders.remove(folder_path)
                
                # process the folder
                self.processor.process_folder(folder_path)
            except Exception as e:
                logging.error(f"error processing modified folder {folder_path}: {e}")
                logging.error(traceback.format_exc())
            finally:
                # always clean up processing set
                self.processing_set.discard(folder_path)

def main():
    args = parse_args()

    # check directories
    if not os.path.exists(args.watch_dir):
        logging.error(f"Watch directory missing: {args.watch_dir}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = PDFProcessor(args.watch_dir, args.output_dir, dpi=args.dpi)

    # process existing folders first
    existing_folders = [os.path.join(args.watch_dir, folder) 
                       for folder in os.listdir(args.watch_dir) 
                       if os.path.isdir(os.path.join(args.watch_dir, folder))]

    if args.max_workers > 1:
        # use threads for parallel processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(processor.process_folder, folder) for folder in existing_folders]
            for future in futures:
                try:
                    future.result()  # wait for completion
                except Exception as e:
                    logging.error(f"Error processing folder: {str(e)}")
    else:
        # process one by one
        for folder in existing_folders:
            processor.process_folder(folder)

    # setup watcher
    event_handler = FolderHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, args.watch_dir, recursive=False)  # non-recursive to avoid nested folders
    observer.start()

    try:
        logging.info(f"Watching for folders in {args.watch_dir}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    # final summary
    logging.info(f"Processing complete. Successes: {processor.success_count} | Failures: {processor.failure_count}")

if __name__ == "__main__":
    main()