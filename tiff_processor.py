import os
import sys
import time
import shutil
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import fitz  # PyMuPDF
from PIL import Image, ImageFile
import numpy as np 
import cv2
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
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
        """convert one pdf page to tiff with opencv adaptive thresholding and group4 compression"""
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
            
            # use opencv for fast adaptive thresholding
            # read image with opencv (grayscale mode)
            img = cv2.imread(temp_png, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image with OpenCV: {temp_png}")
                
            # apply adaptive thresholding (much faster than numpy version)
            binary = cv2.adaptiveThreshold(
                img, 
                255,  # max value
                cv2.ADAPTIVE_THRESH_MEAN_C,  # mean thresholding
                cv2.THRESH_BINARY,  # binary output
                15,  # block size (keep same as before, must be odd)
                5    # c value - constant subtracted from mean
            )
            
            # convert to pil for saving with group 4 compression
            binary_img = Image.fromarray(binary).convert('1')
            
            # save with group4 compression
            binary_img.save(
                output_path, 
                "TIFF", 
                compression="group4", 
                dpi=(self.dpi, self.dpi)
            )
            
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

    def _convert_jpeg_to_tiff(self, jpeg_path, output_path):
        """convert a jpeg file to tiff using opencv adaptive thresholding with group4 compression"""
        try:
            # load image directly with opencv in grayscale mode
            img = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to load image with OpenCV: {jpeg_path}")
            
            # apply adaptive thresholding (much faster implementation)
            binary = cv2.adaptiveThreshold(
                img,
                255,  # max value
                cv2.ADAPTIVE_THRESH_MEAN_C,  # mean thresholding
                cv2.THRESH_BINARY,  # binary output
                15,  # block size (must be odd)
                5    # c value - constant subtracted from mean
            )
            
            # convert to pil for saving with group 4
            binary_img = Image.fromarray(binary).convert('1')
            
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
        """process a single jpeg file to tiff - must succeed"""
        try:
            logging.info(f"processing jpeg: {jpeg_path}")
            
            # setup output path
            base_name = os.path.splitext(os.path.basename(jpeg_path))[0]
            output_dir = os.path.dirname(jpeg_path)
            output_path = os.path.join(output_dir, f"{base_name}.tif")
            
            # try conversion with retries
            max_retries = 5
            success = False
            
            for retry in range(max_retries):
                if retry > 0:
                    logging.info(f"retry attempt {retry} for jpeg: {jpeg_path}")
                    
                try:
                    success = self._convert_jpeg_to_tiff(jpeg_path, output_path)
                    
                    # verify the file exists and is valid
                    if success and os.path.exists(output_path) and self._validate_tiff(output_path):
                        # conversion worked, break out of retry loop
                        break
                    else:
                        # sleep briefly before retrying
                        time.sleep(1)
                except Exception as e:
                    logging.error(f"attempt {retry+1} failed: {str(e)}")
                    # sleep before retry
                    time.sleep(1)
            
            # after all retries, check final status
            if success and os.path.exists(output_path) and self._validate_tiff(output_path):
                # conversion succeeded
                self.success_count += 1
                logging.info(f"successfully converted jpeg: {jpeg_path}")
                self._safe_remove(jpeg_path)
                return True
            else:
                # all retries failed - this is an unrecoverable error
                self.failure_count += 1
                logging.error(f"CRITICAL ERROR: all conversion attempts failed for jpeg: {jpeg_path}")
                # don't delete original (though we'll never move the folder so it will get retried later)
                return False
                    
        except Exception as e:
            self.failure_count += 1
            logging.error(f"CRITICAL ERROR: jpeg processing failed for {jpeg_path}: {str(e)}")
            return False

    def process_pdf(self, pdf_path):
        """convert pdf file to tiff files - must convert ALL pages"""
        success = True
        created_files = []
        doc = None
        
        try:
            # open the pdf file
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.dirname(pdf_path)

            logging.info(f"processing pdf: {pdf_path} ({total_pages} pages)")
            
            # track page conversion status
            all_pages_converted = True
            
            # process each page with retries if needed
            for page_num in range(total_pages):
                output_path = os.path.join(output_dir, f"{base_name}_page_{page_num+1:04d}.tif")
                page_success = False
                
                # try up to 5 times for each page
                max_retries = 5
                for retry in range(max_retries):
                    if retry > 0:
                        logging.info(f"retry attempt {retry} for page {page_num+1}")
                    
                    try:
                        # use lock for thread safety
                        with self.lock:
                            # process the page
                            page_success = self._convert_page(doc[page_num], output_path)
                            
                            if page_success and os.path.exists(output_path) and self._validate_tiff(output_path):
                                # page converted successfully
                                created_files.append(output_path)
                                self.success_count += 1
                                logging.info(f"successfully converted page {page_num+1} of {total_pages}")
                                break  # exit retry loop
                            else:
                                # failed this attempt, will retry
                                logging.warning(f"failed attempt {retry+1} for page {page_num+1}")
                                time.sleep(1)  # brief pause before retry
                    except Exception as e:
                        logging.error(f"error on attempt {retry+1} for page {page_num+1}: {str(e)}")
                        time.sleep(1)  # brief pause before retry
                
                # after all retries, check if we succeeded with this page
                if not page_success or not os.path.exists(output_path) or not self._validate_tiff(output_path):
                    all_pages_converted = False
                    logging.error(f"CRITICAL ERROR: all conversion attempts failed for page {page_num+1}")
                    self.failure_count += 1
            
            # close the document to release file handles
            if doc:
                doc.close()
                doc = None  # Explicitly set to None
            
            # force garbage collection to help release handles
            gc.collect()
            
            # final status check
            if all_pages_converted and len(created_files) == total_pages:
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
            logging.error(f"CRITICAL ERROR: pdf processing failed for {pdf_path}: {str(e)}")
            return False
                
        finally:
            # always close the pdf to avoid memory leaks
            if doc:
                try:
                    doc.close()
                except Exception as e:
                    logging.warning(f"error closing pdf document: {str(e)}")
                    
    def process_folder(self, folder_path):
        """process all pdfs and jpegs in a folder"""
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
                
            # remove from stable folders after processing
            self.stable_folders.discard(folder_path)
                
            return all_successful

        except Exception as e:
            logging.error(f"Folder processing failed: {str(e)}")
            return False

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

    def _move_folder(self, src_folder):
        """move processed folder to output location, merging if destination already exists"""
        try:
            dest_folder = os.path.join(self.output_directory, os.path.basename(src_folder))
            
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