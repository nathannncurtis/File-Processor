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
import gc  # for garbage collection
import psutil  # for checking file handles (pip install psutil)

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

def is_file_stable(file_path, min_wait_time=5, check_interval=1, max_checks=30):
    """check if a file has finished transferring by watching size and mtime"""
    if not os.path.exists(file_path):
        logging.warning(f"file does not exist: {file_path}")
        return False
        
    # get initial state
    try:
        initial_size = os.path.getsize(file_path)
        initial_mtime = os.path.getmtime(file_path)
    except Exception as e:
        logging.warning(f"can't access file {file_path}: {str(e)}")
        return False
    
    # wait minimum time first
    logging.info(f"waiting minimum {min_wait_time} seconds for file stability: {file_path}")
    time.sleep(min_wait_time)
    
    # check if changed after min wait
    try:
        current_size = os.path.getsize(file_path)
        current_mtime = os.path.getmtime(file_path)
        
        if current_size == initial_size and current_mtime == initial_mtime:
            # one extra check to be sure
            time.sleep(check_interval)
            final_size = os.path.getsize(file_path)
            final_mtime = os.path.getmtime(file_path)
            
            if final_size == current_size and final_mtime == current_mtime:
                logging.info(f"file {file_path} is stable after min wait")
                return True
    except Exception as e:
        logging.warning(f"can't check file during stability check: {str(e)}")
        return False
    
    # keep checking if still changing
    prev_size = current_size
    prev_mtime = current_mtime
    stable_count = 0
    needed_stable_checks = 3  # need 3 stable checks in a row
    
    for _ in range(max_checks):
        time.sleep(check_interval)
        
        try:
            current_size = os.path.getsize(file_path)
            current_mtime = os.path.getmtime(file_path)
            
            if current_size == prev_size and current_mtime == prev_mtime:
                stable_count += 1
                logging.debug(f"file stable check {stable_count}/{needed_stable_checks}")
                
                if stable_count >= needed_stable_checks:
                    logging.info(f"file {file_path} is stable after {stable_count} checks")
                    return True
            else:
                # reset counter if changes
                logging.debug(f"file still changing: {file_path}")
                stable_count = 0
                prev_size = current_size
                prev_mtime = current_mtime
                
        except Exception as e:
            logging.warning(f"can't check file stability: {str(e)}")
            return False
    
    logging.warning(f"file {file_path} not stable after {max_checks} checks")
    return False

def is_folder_stable(folder_path, min_wait_time=5, check_interval=1, max_checks=30):
    """check if a folder and all files have finished transferring"""
    if not os.path.exists(folder_path):
        logging.warning(f"folder does not exist: {folder_path}")
        return False
    
    logging.info(f"checking stability for folder: {folder_path}")
    
    # wait minimum time first
    logging.info(f"waiting minimum {min_wait_time} seconds before folder stability check")
    time.sleep(min_wait_time)
    
    # get initial state
    try:
        initial_state = _get_folder_state(folder_path)
        if initial_state is None:
            return False
    except Exception as e:
        logging.error(f"error getting folder state: {str(e)}")
        return False
    
    # check for stability
    stable_count = 0
    needed_stable_checks = 3  # need 3 stable checks in a row
    prev_state = initial_state
    
    for _ in range(max_checks):
        time.sleep(check_interval)
        
        try:
            current_state = _get_folder_state(folder_path)
            if current_state is None:
                return False
                
            if _states_equal(prev_state, current_state):
                stable_count += 1
                logging.debug(f"folder stable check {stable_count}/{needed_stable_checks}")
                
                if stable_count >= needed_stable_checks:
                    logging.info(f"folder {folder_path} is stable after {stable_count} checks")
                    return True
            else:
                # reset if folder changes
                logging.debug(f"folder still changing: {folder_path}")
                stable_count = 0
                prev_state = current_state
                
        except Exception as e:
            logging.error(f"error checking folder stability: {str(e)}")
            return False
    
    logging.warning(f"folder {folder_path} not stable after {max_checks} checks")
    return False

def _get_folder_state(folder_path):
    """get folder state with files, sizes, and mtimes"""
    try:
        if not os.path.exists(folder_path):
            logging.warning(f"folder does not exist: {folder_path}")
            return None
            
        state = {
            'files': {},
            'total_size': 0,
            'count': 0,
        }
        
        for root, _, files in os.walk(folder_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    size = os.path.getsize(file_path)
                    mtime = os.path.getmtime(file_path)
                    state['files'][file_path] = {
                        'size': size,
                        'mtime': mtime
                    }
                    state['total_size'] += size
                    state['count'] += 1
                except Exception as e:
                    logging.warning(f"can't check file {file_path}: {str(e)}")
        
        return state
    except Exception as e:
        logging.error(f"error getting folder state: {str(e)}")
        return None

def _states_equal(state1, state2):
    """compare folder states to see if identical"""
    if state1 is None or state2 is None:
        return False
        
    # check basic stats
    if (state1['count'] != state2['count'] or
        state1['total_size'] != state2['total_size']):
        return False
        
    # check all files same with same sizes and mtimes
    if set(state1['files'].keys()) != set(state2['files'].keys()):
        return False
        
    for path, info1 in state1['files'].items():
        info2 = state2['files'].get(path)
        if info2 is None or info1['size'] != info2['size'] or info1['mtime'] != info2['mtime']:
            return False
            
    return True

def parse_args():
    parser = argparse.ArgumentParser(description="JPEG Processor")
    parser.add_argument('--watch-dir', required=True, help='Directory to watch for new folders containing PDFs')
    parser.add_argument('--output-dir', required=True, help='Directory to move completed folders')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum number of worker processes (use 1 for sequential processing)')
    return parser.parse_args()

def safe_remove_file(file_path, max_retries=5, retry_delay=1):
    """
    try to delete a file, with retries if it's locked
    """
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Successfully removed file on attempt {attempt+1}: {file_path}")
                return True
            else:
                logging.warning(f"File does not exist: {file_path}")
                return True  # already gone, so it's fine
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} to remove file failed: {file_path}, Error: {str(e)}")
            
            if attempt < max_retries - 1:
                # try to figure out what's wrong
                try:
                    if os.path.exists(file_path):
                        # see if we have it open
                        process = psutil.Process(os.getpid())
                        open_files = process.open_files()
                        file_in_use = any(f.path == os.path.abspath(file_path) for f in open_files)
                        if file_in_use:
                            logging.warning(f"File is still in use by this process: {file_path}")
                            # try to force cleanup
                            gc.collect()
                except Exception as diag_error:
                    logging.warning(f"Error diagnosing file lock: {str(diag_error)}")
                
                # pause before retry
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
        """turn pdf into jpg images. returns (converted files list, success)"""
        doc = None
        converted_pages = []
        
        try:
            # check if pdf is stable before opening
            if not is_file_stable(pdf_path, min_wait_time=5, check_interval=1, max_checks=30):
                logging.warning(f"pdf file {pdf_path} not stable after waiting, trying anyway")
            
            # open the pdf
            logging.info(f"opening pdf file: {pdf_path}")
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logging.info(f"Processing {total_pages} pages in PDF: {pdf_path}")

            # prep output paths
            pdf_dir = os.path.dirname(pdf_path)
            pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            
            # process pages
            successful_pages = 0
            for page_num in range(total_pages):
                try:
                    # grab page
                    page = doc[page_num]
                    
                    # make pixmap at 200 dpi
                    pix = page.get_pixmap(dpi=200)
                    
                    # setup output name
                    output_filename = f"{pdf_filename}_page_{page_num + 1:04d}.jpg"
                    output_path = os.path.join(pdf_dir, output_filename)
                    
                    # convert pixmap to PIL Image
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    
                    # store dpi in image
                    img.info['dpi'] = (200, 200)
                    
                    # save as jpeg
                    img.save(output_path, "JPEG", quality=100, dpi=(200, 200))
                    
                    logging.info(f"Saved page {page_num + 1} as {output_filename}")
                    converted_pages.append(output_path)
                    successful_pages += 1
                
                except Exception as page_error:
                    logging.error(f"Error processing page {page_num + 1} of {pdf_path}: {page_error}")
            
            # check if all pages worked
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
            # cleanup pdf
            if doc:
                try:
                    doc.close()
                    doc = None  # help garbage collection
                except Exception as close_error:
                    logging.error(f"Error closing PDF {pdf_path}: {close_error}")

    def process_folder(self, folder_path):
        """process all pdfs in a folder"""
        logging.info(f"Processing folder: {folder_path}")
        
        try:
            # check if folder exists
            if not os.path.exists(folder_path):
                logging.error(f"Folder does not exist: {folder_path}")
                return False
            
            # check folder stability
            if not is_folder_stable(folder_path, min_wait_time=5, check_interval=1, max_checks=30):
                logging.warning(f"folder {folder_path} not stable after waiting, trying anyway")
                
            # get all files
            files = os.listdir(folder_path)
            
            # find pdfs
            pdf_files = [f for f in files if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                logging.info(f"No PDF files found in {folder_path}")
                return False
            
            # process each pdf
            for pdf_filename in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_filename)
                
                # convert to jpegs
                converted_pages, all_pages_converted = self.process_pdf_to_jpgs(pdf_path)
                
                # cleanup memory
                gc.collect()
                
                # wait for file handles to release
                time.sleep(1)
                
                # remove original pdf if successful
                if converted_pages and all_pages_converted:
                    # try to delete original
                    if not safe_remove_file(pdf_path):
                        logging.warning(f"Could not delete PDF {pdf_path}. Manual deletion may be required.")
                else:
                    logging.warning(f"Not deleting PDF {pdf_path} because conversion was incomplete")
            
            # move folder when done
            dest_folder = os.path.join(self.output_directory, os.path.basename(folder_path))
            
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
                    
                    shutil.move(src_item, dest_item)
                
                # remove source folder
                try:
                    os.rmdir(folder_path)
                    logging.info(f"Merged contents of {folder_path} into {dest_folder}")
                except Exception as rmdir_error:
                    logging.error(f"Error removing source folder {folder_path}: {rmdir_error}")
            else:
                # just move the whole folder
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
        self.processing_set = set()  # track what we're working on

    def on_created(self, event):
        if event.is_directory:
            folder_path = event.src_path
            
            # avoid processing same folder multiple times
            if folder_path in self.processing_set:
                logging.info(f"Folder already being processed: {folder_path}")
                return
                
            logging.info(f"New folder detected: {folder_path}")
            
            # add to processing set right away
            self.processing_set.add(folder_path)
            
            try:
                # check stability before processing
                if not is_folder_stable(folder_path, min_wait_time=5, check_interval=1, max_checks=30):
                    logging.warning(f"folder {folder_path} not stable after waiting, trying anyway")
                
                # process the folder
                self.processor.process_folder(folder_path)
            except Exception as e:
                logging.error(f"error processing folder {folder_path}: {e}")
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
    processor = PDFProcessor(args.watch_dir, args.output_dir)
    
    # process any existing folders first
    for item in os.listdir(args.watch_dir):
        item_path = os.path.join(args.watch_dir, item)
        if os.path.isdir(item_path):
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