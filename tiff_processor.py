import os
import sys
import time
import shutil
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fitz  # PyMuPDF
from PIL import Image, ImageFile
import numpy as np  # Add this import
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

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
        self.lock = threading.Lock()  # lock for thread-safe operations

    def _validate_tiff(self, file_path):
        """check if tiff file is valid and can be opened"""
        try:
            # first check basics - file exists and has size
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                logging.error(f"TIFF file {file_path} is empty or doesn't exist")
                return False
                
            # try opening the file to make sure it's usable
            with Image.open(file_path) as img:
                img.load()  # force load to catch any issues
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

    def _convert_page(self, page, output_path):
        """convert one pdf page to TIFF with adaptive thresholding and group4 compression"""
        temp_png = None
        try:
            # create temp file for intermediate png
            temp_dir = os.path.dirname(output_path)
            temp_basename = f"temp_{os.path.basename(output_path).replace('.tif', '')}_{int(time.time()*1000)}.png"
            temp_png = os.path.join(temp_dir, temp_basename)
            
            # generate pixmap with our dpi settings
            pix = page.get_pixmap(dpi=self.dpi, alpha=False)
            
            # save as png first - easier to work with
            pix.pil_save(temp_png, format="PNG")
            
            # check temp file was created properly
            if not os.path.exists(temp_png) or os.path.getsize(temp_png) == 0:
                raise ValueError(f"Failed to create temporary PNG file: {temp_png}")
            
            # load image with PIL for adaptive thresholding
            with Image.open(temp_png) as img:
                # convert to grayscale
                gray_img = img.convert("L")
                
                # apply adaptive thresholding
                img_array = np.array(gray_img)
                h, w = img_array.shape
                binary = np.zeros_like(img_array)
                
                # process in blocks for speed
                block_size = 15  # local region size
                c = 5  # constant subtracted from mean (controls threshold sensitivity)
                step = max(1, block_size // 2)
                
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        # define block region
                        i_end = min(i + block_size, h)
                        j_end = min(j + block_size, w)
                        block = img_array[i:i_end, j:j_end]
                        
                        # calculate threshold for this block
                        threshold = np.mean(block) - c
                        
                        # apply threshold to the block
                        binary[i:i_end, j:j_end] = np.where(block < threshold, 0, 255)
                
                # convert back to PIL image and ensure 1-bit mode for Group 4
                binary_img = Image.fromarray(binary.astype(np.uint8)).convert('1')
                
                # save with group4 compression
                binary_img.save(
                    output_path, 
                    "TIFF", 
                    compression="group4", 
                    dpi=(self.dpi, self.dpi)
                )
            
            # verify the TIFF is good
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

    def _convert_jpeg_to_tiff(self, jpeg_path, output_path):
        """convert a jpeg file to tiff using adaptive thresholding with group4 compression"""
        try:
            with Image.open(jpeg_path) as img:
                # convert to grayscale
                gray_img = img.convert("L")
                
                # apply adaptive thresholding
                img_array = np.array(gray_img)
                h, w = img_array.shape
                binary = np.zeros_like(img_array)
                
                # process in blocks for speed
                block_size = 15  # local region size
                c = 5  # constant subtracted from mean
                step = max(1, block_size // 2)
                
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        # define block region
                        i_end = min(i + block_size, h)
                        j_end = min(j + block_size, w)
                        block = img_array[i:i_end, j:j_end]
                        
                        # calculate threshold for this block
                        threshold = np.mean(block) - c
                        
                        # apply threshold to the block
                        binary[i:i_end, j:j_end] = np.where(block < threshold, 0, 255)
                
                # convert back to PIL image in 1-bit mode
                binary_img = Image.fromarray(binary.astype(np.uint8)).convert('1')
                
                # save with group4 compression
                binary_img.save(
                    output_path, 
                    "TIFF", 
                    compression="group4", 
                    dpi=(self.dpi, self.dpi)
                )
            
            # make sure the tiff is valid
            if not self._validate_tiff(output_path):
                raise ValueError("TIFF validation failed")
                
            return True
        except Exception as e:
            logging.error(f"JPEG to TIFF conversion failed for {jpeg_path}: {str(e)}")
            if os.path.exists(output_path):
                self._safe_remove(output_path)
            return False
        
    def process_jpeg(self, jpeg_path):
        """process a single jpeg file to tiff"""
        try:
            logging.info(f"Processing JPEG: {jpeg_path}")
            
            # setup output path
            base_name = os.path.splitext(os.path.basename(jpeg_path))[0]
            output_dir = os.path.dirname(jpeg_path)
            output_path = os.path.join(output_dir, f"{base_name}.tif")
            
            # do the conversion
            success = self._convert_jpeg_to_tiff(jpeg_path, output_path)
            
            if success:
                # clean up original jpeg if successful
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
        """convert pdf file to tiff files - one per page"""
        success = True
        created_files = []
        doc = None
        
        try:
            # open the pdf file
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.dirname(pdf_path)

            logging.info(f"Processing PDF: {pdf_path} ({total_pages} pages)")
            
            # process each page
            for page_num in range(total_pages):
                output_path = os.path.join(output_dir, f"{base_name}_page_{page_num+1:04d}.tif")
                
                try:
                    # use lock for thread safety
                    with self.lock:
                        # process the page
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
            
            # only successful if all pages worked
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
            # always close the pdf to avoid memory leaks
            if doc:
                try:
                    doc.close()
                except Exception as e:
                    logging.warning(f"Error closing PDF document: {str(e)}")
                    
            # cleanup based on success
            if success:
                # remove original pdf if all went well
                self._safe_remove(pdf_path)
            else:
                # keep the created files anyway - might be useful
                pass

    def process_folder(self, folder_path):
        """process all pdfs and jpegs in a folder"""
        try:
            # check folder still exists
            if not os.path.exists(folder_path):
                logging.info(f"Folder no longer exists: {folder_path}")
                return False
                
            # see if there's anything to process
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            jpeg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
            
            if not pdf_files and not jpeg_files:
                # nothing to process
                if folder_path in self.processed_folders:
                    logging.info(f"Folder already processed and has no new files: {folder_path}")
                    return True
                else:
                    logging.info(f"No PDF or JPEG files found in folder: {folder_path}")
                    self._move_folder(folder_path)
                    self.processed_folders.add(folder_path)
                    return True
            
            # got files to process
            if folder_path in self.processed_folders:
                logging.info(f"Folder previously processed but contains new files: {folder_path}")
                self.processed_folders.remove(folder_path)
            
            # make sure folder isn't still being written to
            if not self._is_folder_stable(folder_path):
                return False

            # process pdfs
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            pdf_results = []
            
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)
                result = self.process_pdf(pdf_path)
                pdf_results.append(result)

            # process jpegs
            jpeg_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.jpeg'))]
            jpeg_results = []
            
            for jpeg_file in jpeg_files:
                jpeg_path = os.path.join(folder_path, jpeg_file)
                result = self.process_jpeg(jpeg_path)
                jpeg_results.append(result)

            # move folder if everything worked
            all_successful = True
            if pdf_files or jpeg_files:
                if not all(pdf_results + jpeg_results):
                    all_successful = False
                    logging.warning(f"Not all files in {folder_path} were processed successfully")
                    
                if all_successful:
                    self._move_folder(folder_path)
                    self.processed_folders.add(folder_path)  # mark as done
            else:
                # empty folders just get moved
                logging.info(f"No PDF or JPEG files found in {folder_path}")
                self._move_folder(folder_path)
                self.processed_folders.add(folder_path)
                
            return all_successful

        except Exception as e:
            logging.error(f"Folder processing failed: {str(e)}")
            return False

    def _is_folder_stable(self, folder_path, timeout=30, check_interval=2):
        """check if folder is done changing before processing"""
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
            return True  # process anyway after timeout
        except Exception as e:
            logging.error(f"Stability check failed: {str(e)}")
            return False

    def _get_folder_state(self, folder_path):
        """get folder state info to check if it's changing"""
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
        """move processed folder to output location"""
        try:
            dest_folder = os.path.join(self.output_directory, os.path.basename(src_folder))
            
            if os.path.exists(dest_folder):
                logging.warning(f"Destination exists: {dest_folder}")
                # use timestamp to make unique name
                dest_folder = f"{dest_folder}_{int(time.time())}"
                
            # make sure destination folder exists
            os.makedirs(os.path.dirname(dest_folder), exist_ok=True)
            
            # do the move
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
            time.sleep(5)  # give time for file transfers
            # reset if folder gets recreated
            if event.src_path in self.processor.processed_folders:
                self.processor.processed_folders.remove(event.src_path)
            self.processor.process_folder(event.src_path)

    def on_modified(self, event):
        if event.is_directory:
            logging.info(f"Folder modified: {event.src_path}")
            time.sleep(5)  # give time for file transfers
            # check for files before processing
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