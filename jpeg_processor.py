import os
import sys
import time
import shutil
import uuid
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fitz  # PyMuPDF
from PIL import Image
import io
import argparse
import logging
import traceback
import gc  # For garbage collection
import psutil  # For checking file handles (pip install psutil)

# Determine the script directory for absolute paths
if getattr(sys, 'frozen', False):
    # If running as a frozen exe
    script_dir = os.path.dirname(sys.executable)
else:
    # If running as a regular Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

# Create log file with absolute path
log_file = os.path.join(script_dir, "jpeg_processor.log")
print(f"Using log file: {log_file}")  # Print to console for debugging

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # Also log to console
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="JPEG Processor")
    parser.add_argument('--watch-dir', required=True, help='Directory to watch for new folders containing PDFs')
    parser.add_argument('--output-dir', required=True, help='Directory to move completed folders')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum number of worker processes (use 1 for sequential processing)')
    return parser.parse_args()

def safe_remove_file(file_path, max_retries=5, retry_delay=1):
    """
    Attempt to safely remove a file with retries.
    Returns True if successful, False otherwise.
    """
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Successfully removed file on attempt {attempt+1}: {file_path}")
                return True
            else:
                logging.warning(f"File does not exist: {file_path}")
                return True  # File doesn't exist, so consider removal successful
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} to remove file failed: {file_path}, Error: {str(e)}")
            
            if attempt < max_retries - 1:
                # Try to diagnose the issue
                try:
                    if os.path.exists(file_path):
                        # Check if file is locked
                        process = psutil.Process(os.getpid())
                        open_files = process.open_files()
                        file_in_use = any(f.path == os.path.abspath(file_path) for f in open_files)
                        if file_in_use:
                            logging.warning(f"File is still in use by this process: {file_path}")
                            # Force garbage collection to try to release file handles
                            gc.collect()
                except Exception as diag_error:
                    logging.warning(f"Error diagnosing file lock: {str(diag_error)}")
                
                # Wait before retrying
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to remove file after {max_retries} attempts: {file_path}")
                return False
    return False

class PDFProcessor:
    def __init__(self, watch_directory, output_directory):
        self.watch_directory = watch_directory
        self.output_directory = output_directory
        logging.info(f"Initializing PDFProcessor with watch dir: {watch_directory}, output dir: {output_directory}")

    def process_pdf_to_jpgs(self, pdf_path):
        """Convert a PDF to JPEG images. Returns (converted_pages, success_status)"""
        doc = None
        converted_pages = []
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logging.info(f"Processing {total_pages} pages in PDF: {pdf_path}")

            # Prepare output directory (same as PDF location)
            pdf_dir = os.path.dirname(pdf_path)
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # Process each page
            successful_pages = 0
            for page_num in range(total_pages):
                try:
                    # Extract page
                    page = doc[page_num]
                    
                    # Create pixmap with 200 DPI
                    pix = page.get_pixmap(dpi=200)
                    
                    # Prepare output filename
                    output_filename = f"{pdf_filename}_page_{page_num + 1:04d}.jpg"
                    output_path = os.path.join(pdf_dir, output_filename)
                    
                    # Convert pixmap to PIL Image and preserve DPI information
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    
                    # Ensure we explicitly set the DPI in the image object 
                    img.info['dpi'] = (200, 200)
                    
                    # Save as JPEG with explicit DPI metadata
                    img.save(output_path, "JPEG", quality=100, dpi=(200, 200))
                    
                    logging.info(f"Saved page {page_num + 1} as {output_filename}")
                    converted_pages.append(output_path)
                    successful_pages += 1
                
                except Exception as page_error:
                    logging.error(f"Error processing page {page_num + 1} of {pdf_path}: {page_error}")
            
            # Check if all pages were successfully converted
            all_pages_converted = (successful_pages == total_pages)
            if all_pages_converted:
                logging.info(f"Successfully converted all {total_pages} pages from {pdf_path}")
            else:
                logging.warning(f"Only converted {successful_pages} out of {total_pages} pages from {pdf_path}")
            
            return converted_pages, all_pages_converted
        
        except Exception as e:
            logging.error(f"Error processing PDF {pdf_path}: {e}")
            logging.error(traceback.format_exc())
            return converted_pages, False
        
        finally:
            # Properly close the PDF to release file handles
            if doc:
                try:
                    doc.close()
                    doc = None  # Explicitly set to None to help garbage collection
                except Exception as close_error:
                    logging.error(f"Error closing PDF {pdf_path}: {close_error}")

    def process_folder(self, folder_path):
        """Process all PDFs in a folder."""
        logging.info(f"Processing folder: {folder_path}")
        
        try:
            # Make sure the folder exists
            if not os.path.exists(folder_path):
                logging.error(f"Folder does not exist: {folder_path}")
                return False
                
            # List all files in the folder
            files = os.listdir(folder_path)
            
            # Filter PDF files
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logging.info(f"No PDF files found in {folder_path}")
                return False
            
            # Process each PDF
            for pdf_filename in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_filename)
                
                # Convert PDF to JPEGs
                converted_pages, all_pages_converted = self.process_pdf_to_jpgs(pdf_path)
                
                # Force garbage collection to release file handles
                gc.collect()
                
                # Wait a moment to ensure file handles are released
                time.sleep(1)
                
                # If conversion successful, remove the original PDF
                if converted_pages and all_pages_converted:
                    # Try to safely remove the PDF
                    if not safe_remove_file(pdf_path):
                        logging.warning(f"Could not delete PDF {pdf_path}. Manual deletion may be required.")
                else:
                    logging.warning(f"Not deleting PDF {pdf_path} because conversion was incomplete")
            
            # Move the folder to completed directory
            dest_folder = os.path.join(self.output_directory, os.path.basename(folder_path))
            
            # If destination exists, merge contents
            if os.path.exists(dest_folder):
                for item in os.listdir(folder_path):
                    src_item = os.path.join(folder_path, item)
                    dest_item = os.path.join(dest_folder, item)
                    
                    # If item already exists in destination, rename
                    if os.path.exists(dest_item):
                        base, ext = os.path.splitext(item)
                        counter = 1
                        while os.path.exists(dest_item):
                            new_filename = f"{base}_{counter}{ext}"
                            dest_item = os.path.join(dest_folder, new_filename)
                            counter += 1
                    
                    shutil.move(src_item, dest_item)
                
                # Remove the source folder
                try:
                    os.rmdir(folder_path)
                    logging.info(f"Merged contents of {folder_path} into {dest_folder}")
                except Exception as rmdir_error:
                    logging.error(f"Error removing source folder {folder_path}: {rmdir_error}")
            else:
                # Simply move the folder
                shutil.move(folder_path, dest_folder)
                logging.info(f"Moved folder: {folder_path} -> {dest_folder}")
            
            return True
        
        except Exception as e:
            logging.error(f"Error processing folder {folder_path}: {e}")
            logging.error(traceback.format_exc())
            return False

class FolderWatcher(FileSystemEventHandler):
    def __init__(self, processor):
        self.processor = processor
        self.processing_set = set()  # Track folders being processed

    def on_created(self, event):
        if event.is_directory:
            folder_path = event.src_path
            
            # Avoid processing the same folder multiple times simultaneously
            if folder_path in self.processing_set:
                logging.info(f"Folder already being processed: {folder_path}")
                return
                
            logging.info(f"New folder detected: {folder_path}")
            
            # Wait a bit to ensure folder is fully created
            time.sleep(2)
            
            # Mark as being processed
            self.processing_set.add(folder_path)
            
            try:
                self.processor.process_folder(folder_path)
            finally:
                # Always remove from processing set when done
                self.processing_set.discard(folder_path)

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Validate directories
    if not os.path.exists(args.watch_dir):
        logging.error(f"Watch directory does not exist: {args.watch_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")
    
    # Create processor
    processor = PDFProcessor(args.watch_dir, args.output_dir)
    
    # Process any existing folders first
    for item in os.listdir(args.watch_dir):
        item_path = os.path.join(args.watch_dir, item)
        if os.path.isdir(item_path):
            processor.process_folder(item_path)
    
    # Set up file watcher
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