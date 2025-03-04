import os
import sys
import time
import shutil
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fitz  # PyMuPDF
from PIL import Image, ImageFile
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure PIL to be more tolerant of damaged files
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tiff_processor.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

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
        self.processed_folders = set()  # Track processed folders
        self.lock = threading.Lock()  # Lock for thread-safe file operations

    def _validate_tiff(self, file_path):
        """Ensure TIFF file is valid and can be opened"""
        try:
            # First check if file exists and has size > 0
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logging.error(f"TIFF file {file_path} is empty or doesn't exist")
                return False
                
            # Try to open and verify the TIFF
            with Image.open(file_path) as img:
                img.load()  # Verify the file can be loaded
                # Make sure it's a valid TIFF format
                if img.format != "TIFF":
                    logging.error(f"File {file_path} is not a valid TIFF format")
                    return False
            return True
        except Exception as e:
            logging.error(f"Invalid TIFF file {file_path}: {str(e)}")
            return False

    def _safe_remove(self, file_path, max_retries=3, retry_delay=1):
        """Safely remove a file with retries, skipping if it's locked or doesn't exist"""
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

    def _convert_page(self, page, output_path):
        """Convert single PDF page to TIFF with CCITT Group 4 compression"""
        temp_png = None
        try:
            # Create a temporary file path for the PNG with unique name based on timestamp
            temp_dir = os.path.dirname(output_path)
            temp_basename = f"temp_{os.path.basename(output_path).replace('.tif', '')}_{int(time.time()*1000)}.png"
            temp_png = os.path.join(temp_dir, temp_basename)
            
            # Generate pixmap with specified DPI (use monochrome rendering for better results with Group 4)
            pix = page.get_pixmap(dpi=self.dpi, alpha=False)
            
            # Save as uncompressed PNG for intermediate processing
            pix.pil_save(temp_png, format="PNG")
            
            # Make sure temp file was created
            if not os.path.exists(temp_png) or os.path.getsize(temp_png) == 0:
                raise ValueError(f"Failed to create temporary PNG file: {temp_png}")
            
            # Convert to TIFF using PIL with proper handling for Group 4 compression
            with Image.open(temp_png) as img:
                # Convert to binary (1-bit) with threshold
                # This is critical for Group 4 compression which requires 1-bit images
                binary_img = img.convert("L").point(lambda x: 0 if x < 128 else 255, '1')
                
                # Save with explicit Group 4 compression and resolution
                binary_img.save(
                    output_path, 
                    "TIFF", 
                    compression="group4", 
                    dpi=(self.dpi, self.dpi)
                )
            
            # Verify the TIFF is valid and can be opened
            if not self._validate_tiff(output_path):
                raise ValueError(f"TIFF validation failed for {output_path}")
                
            return True
        except Exception as e:
            logging.error(f"PDF page conversion failed: {str(e)}")
            # Clean up any partial output
            if os.path.exists(output_path):
                self._safe_remove(output_path)
            return False
        finally:
            # Always clean up the temporary file
            if temp_png and os.path.exists(temp_png):
                self._safe_remove(temp_png)

    def _convert_jpeg_to_tiff(self, jpeg_path, output_path):
        """Convert JPEG to TIFF with CCITT Group 4 compression"""
        try:
            with Image.open(jpeg_path) as img:
                # Convert to grayscale
                gray_img = img.convert("L")
                # Apply threshold to get binary image
                binary_img = gray_img.point(lambda x: 0 if x < 128 else 255, '1')
                # Save as TIFF with CCITT Group 4 compression
                binary_img.save(
                    output_path, 
                    "TIFF", 
                    compression="group4", 
                    dpi=(self.dpi, self.dpi)
                )
            
            # Verify the TIFF is valid
            if not self._validate_tiff(output_path):
                raise ValueError("TIFF validation failed")
                
            return True
        except Exception as e:
            logging.error(f"JPEG to TIFF conversion failed for {jpeg_path}: {str(e)}")
            if os.path.exists(output_path):
                self._safe_remove(output_path)
            return False

    def process_jpeg(self, jpeg_path):
        """Process a single JPEG file to TIFF"""
        try:
            logging.info(f"Processing JPEG: {jpeg_path}")
            
            # Create output path
            base_name = os.path.splitext(os.path.basename(jpeg_path))[0]
            output_dir = os.path.dirname(jpeg_path)
            output_path = os.path.join(output_dir, f"{base_name}.tif")
            
            # Convert JPEG to TIFF
            success = self._convert_jpeg_to_tiff(jpeg_path, output_path)
            
            if success:
                # Remove original JPEG after successful conversion
                self.success_count += 1
                logging.info(f"Successfully converted JPEG: {jpeg_path}")
                self._safe_remove(jpeg_path)
                return True
            else:
                self.failure_count += 1
                logging.warning(f"Failed to convert JPEG: {jpeg_path}")
                return False
                
        except Exception as e:
            self.failure_count += 1
            logging.error(f"JPEG processing failed for {jpeg_path}: {str(e)}")
            return False

    def process_pdf(self, pdf_path):
        """Convert PDF to validated TIFF files"""
        success = True
        created_files = []
        doc = None
        
        try:
            # Open the PDF with proper error handling
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.dirname(pdf_path)

            logging.info(f"Processing PDF: {pdf_path} ({total_pages} pages)")
            
            # Process each page with proper error handling
            for page_num in range(total_pages):
                output_path = os.path.join(output_dir, f"{base_name}_page_{page_num+1:04d}.tif")
                
                try:
                    # Use a lock for thread-safe file operations
                    with self.lock:
                        # Process the page
                        if self._convert_page(doc[page_num], output_path):
                            created_files.append(output_path)
                            self.success_count += 1
                            logging.info(f"Successfully converted page {page_num+1} of {total_pages}")
                        else:
                            success = False
                            self.failure_count += 1
                            logging.warning(f"Failed to convert page {page_num+1} of {total_pages}")
                except Exception as e:
                    success = False
                    self.failure_count += 1
                    logging.error(f"Error processing page {page_num+1}: {str(e)}")
            
            # Only consider success if all pages were converted
            if success and len(created_files) == total_pages:
                logging.info(f"All {total_pages} pages of {pdf_path} processed successfully")
            else:
                success = False
                logging.warning(f"Only {len(created_files)} of {total_pages} pages processed for {pdf_path}")
                
            return success
            
        except Exception as e:
            success = False
            logging.error(f"PDF processing failed for {pdf_path}: {str(e)}")
            return False
            
        finally:
            # Always close the document to prevent memory leaks
            if doc:
                try:
                    doc.close()
                except Exception as e:
                    logging.warning(f"Error closing PDF document: {str(e)}")
                    
            # Handle cleanup based on success state
            if success:
                # Remove the original PDF if successful
                self._safe_remove(pdf_path)
            else:
                # Leave the created files as they may still be useful
                pass

    def process_folder(self, folder_path):
        """Process all PDFs and JPEGs in a folder with validation"""
        try:
            # First check if the folder exists - it might have been moved already
            if not os.path.exists(folder_path):
                logging.info(f"Folder no longer exists: {folder_path}")
                return False
                
            # Check if folder has processable files before deciding to skip
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
            
            if not pdf_files and not jpeg_files:
                # No PDF or JPEG files to process
                if folder_path in self.processed_folders:
                    logging.info(f"Folder already processed and has no new files: {folder_path}")
                    return True
                else:
                    logging.info(f"No PDF or JPEG files found in folder: {folder_path}")
                    self._move_folder(folder_path)
                    self.processed_folders.add(folder_path)
                    return True
            
            # If we found processable files, always process the folder
            # This allows reprocessing when new files are added
            if folder_path in self.processed_folders:
                logging.info(f"Folder previously processed but contains new files: {folder_path}")
                self.processed_folders.remove(folder_path)
            
            # Validate folder stability
            if not self._is_folder_stable(folder_path):
                return False

            # Process PDFs
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            pdf_results = []
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)
                result = self.process_pdf(pdf_path)
                pdf_results.append(result)

            # Process JPEGs
            jpeg_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg'))]
            jpeg_results = []
            
            for jpeg_file in jpeg_files:
                jpeg_path = os.path.join(folder_path, jpeg_file)
                result = self.process_jpeg(jpeg_path)
                jpeg_results.append(result)

            # Only move folder if all files were processed successfully
            all_successful = True
            if pdf_files or jpeg_files:
                if not all(pdf_results + jpeg_results):
                    all_successful = False
                    logging.warning(f"Not all files in {folder_path} were processed successfully")
                    
                if all_successful:
                    self._move_folder(folder_path)
                    self.processed_folders.add(folder_path)  # Mark folder as processed
            else:
                # Handle empty folders
                logging.info(f"No PDF or JPEG files found in {folder_path}")
                self._move_folder(folder_path)
                self.processed_folders.add(folder_path)  # Mark folder as processed
                
            return all_successful

        except Exception as e:
            logging.error(f"Folder processing failed: {str(e)}")
            return False

    def _is_folder_stable(self, folder_path, timeout=30, check_interval=2):
        """Ensure folder has finished receiving files"""
        logging.info(f"Verifying folder stability: {folder_path}")
        try:
            initial_state = None
            start_time = time.time()

            while time.time() - start_time < timeout:
                if not os.path.exists(folder_path):
                    logging.warning(f"Folder no longer exists: {folder_path}")
                    return False

                current_state = self._get_folder_state(folder_path)
                if current_state == initial_state and current_state is not None:
                    return True
                initial_state = current_state
                time.sleep(check_interval)

            logging.warning(f"Folder not stable after {timeout} seconds, processing anyway")
            return True  # Process anyway after timeout
        except Exception as e:
            logging.error(f"Stability check failed: {str(e)}")
            return False

    def _get_folder_state(self, folder_path):
        """Get folder state signature"""
        try:
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Folder not found: {folder_path}")

            files = os.listdir(folder_path)
            return {
                'file_count': len(files),
                'file_list': sorted(files),
                'total_size': sum(os.path.getsize(os.path.join(folder_path, f)) 
                                for f in files if os.path.isfile(os.path.join(folder_path, f)))
            }
        except Exception as e:
            logging.error(f"Failed to get folder state: {str(e)}")
            return None

    def _move_folder(self, src_folder):
        """Safely move processed folder"""
        try:
            dest_folder = os.path.join(self.output_directory, os.path.basename(src_folder))
            
            if os.path.exists(dest_folder):
                logging.warning(f"Destination exists: {dest_folder}")
                # Use a timestamp to create a unique folder name
                dest_folder = f"{dest_folder}_{int(time.time())}"
                
            # Ensure destination parent directory exists
            os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
            
            # Move the folder
            shutil.move(src_folder, dest_folder)
            logging.info(f"Moved folder: {src_folder} -> {dest_folder}")
            return True
        except Exception as e:
            logging.error(f"Folder move failed: {str(e)}")
            return False

class FolderHandler(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor

    def on_created(self, event):
        if event.is_directory:
            logging.info(f"New folder detected: {event.src_path}")
            time.sleep(5)  # Allow time for file transfers
            # Always remove from processed set when a new folder is created
            # This allows reprocessing of folders with the same name
            if event.src_path in self.processor.processed_folders:
                self.processor.processed_folders.remove(event.src_path)
            self.processor.process_folder(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            logging.info(f"Folder modified: {event.src_path}")
            time.sleep(5)  # Allow time for file transfers
            # Check if the folder has processable files before processing
            if os.path.exists(event.src_path):
                try:
                    pdf_files = [f for f in os.listdir(event.src_path) if f.lower().endswith('.pdf')]
                    jpeg_files = [f for f in os.listdir(event.src_path) if f.lower().endswith(('.jpg', '.jpeg'))]
                    if pdf_files or jpeg_files:
                        self.processor.process_folder(event.src_path)
                    else:
                        logging.info(f"Folder modified but contains no processable files: {event.src_path}")
                except Exception as e:
                    logging.error(f"Error checking folder contents: {str(e)}")
            else:
                logging.info(f"Modified folder no longer exists: {event.src_path}")

def main():
    args = parse_args()

    # Validate directories
    if not os.path.exists(args.watch_dir):
        logging.error(f"Watch directory missing: {args.watch_dir}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = PDFProcessor(args.watch_dir, args.output_dir, dpi=args.dpi)

    # Process existing folders first
    existing_folders = [os.path.join(args.watch_dir, folder) 
                       for folder in os.listdir(args.watch_dir) 
                       if os.path.isdir(os.path.join(args.watch_dir, folder))]

    if args.max_workers > 1:
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(processor.process_folder, folder) for folder in existing_folders]
            for future in futures:
                try:
                    future.result()  # Wait for each task to complete
                except Exception as e:
                    logging.error(f"Error processing folder: {str(e)}")
    else:
        # Process folders sequentially
        for folder in existing_folders:
            processor.process_folder(folder)

    # Set up watcher
    event_handler = FolderHandler(processor)
    observer = Observer()
    observer.schedule(event_handler, args.watch_dir, recursive=False)  # Non-recursive to avoid processing nested folders
    observer.start()

    try:
        logging.info(f"Watching for folders in {args.watch_dir}")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    # Final report
    logging.info(f"Processing complete. Successes: {processor.success_count} | Failures: {processor.failure_count}")

if __name__ == "__main__":
    main()