import os
import sys
import json
import shutil
import subprocess
import multiprocessing
import time
import logging
import traceback
import threading

# figure out the script directory for absolute paths
if getattr(sys, 'frozen', False):
    # running as a frozen exe
    script_dir = os.path.dirname(sys.executable)
else:
    # regular python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

# setup log file with absolute path
log_file = os.path.join(script_dir, "job_manager.log")
print(f"Job Manager using log file: {log_file}")

# setup logging to file and console
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# add console logging too
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# log startup info
logging.info(f"Started job_manager")
logging.info(f"Python executable: {sys.executable}")
logging.info(f"Script directory: {script_dir}")
logging.info(f"Current working directory: {os.getcwd()}")

class JobManager:
    def __init__(self, config_file):
        """start up the job manager with config file path"""
        self.config_file = config_file
        self.processes = {}  # track running processes by job
        self.transitioning_profiles = set()  # track profiles changing state
        self._config_changed = False  # Track if config needs saving
        self._last_config_save = time.time()
        self._config_lock = threading.Lock()  # Thread safety for config operations
        
        # File tracking to prevent reprocessing the same files
        self._processed_files = {}  # file_path -> (modification_time, process_time)
        self._file_lock = threading.Lock()  # Thread safety for file tracking
        
        
        self.load_config()
        
        # Start background config saver
        self._start_config_saver()
        
        # Start simple file checker
        self._start_file_checker()
        
        # log init
        logging.info(f"JobManager initialized with config file: {config_file}")
        if not os.path.exists(config_file):
            logging.warning(f"Config file does not exist, will create: {config_file}")

    def _start_config_saver(self):
        """Start background thread to handle config saves"""
        def config_save_worker():
            while True:
                try:
                    time.sleep(2)  # Check every 2 seconds
                    if self._config_changed:
                        current_time = time.time()
                        # Only save if it's been at least 1 second since last save
                        if current_time - self._last_config_save >= 1:
                            with self._config_lock:
                                if self._config_changed:  # Double check inside lock
                                    self._do_config_save()
                                    self._config_changed = False
                                    self._last_config_save = current_time
                except Exception as e:
                    logging.error(f"Error in config save worker: {e}")
        
        config_thread = threading.Thread(target=config_save_worker, daemon=True)
        config_thread.start()
        logging.info("Background config saver thread started")

    def _start_file_checker(self):
        """Start background thread to continuously check for files to process"""
        def file_check_worker():
            while True:
                try:
                    time.sleep(10)  # Check every 10 seconds
                    self._check_all_active_profiles()
                except Exception as e:
                    logging.error(f"Error in file check worker: {e}")
                    logging.error(f"File checker continuing despite error...")
                    # Continue the loop no matter what happens
                    continue
        
        check_thread = threading.Thread(target=file_check_worker, daemon=True)
        check_thread.start()
        logging.info("Background file checker thread started")

    def _check_all_active_profiles(self):
        """Check all active profiles for files to process"""
        try:
            active_profiles = []
            for profile_name, details in self.config.get('profiles', {}).items():
                try:
                    # Only check active profiles
                    if details.get('status') != 'Active':
                        continue
                    
                    active_profiles.append(profile_name)
                    self._scan_profile_folders(profile_name, details)
                    
                except Exception as e:
                    logging.error(f"Error checking profile {profile_name}: {e}")
                    logging.error(f"Continuing with next profile...")
                    continue
            
            # Log checker activity every 60 seconds (every 6th check)
            current_time = time.time()
            if int(current_time) % 60 < 10:  # Log once per minute
                with self._file_lock:
                    tracked_files = len(self._processed_files)
                logging.info(f"File checker: {len(active_profiles)} active profiles - {', '.join(active_profiles)} (tracking {tracked_files} files)")
        except Exception as e:
            logging.error(f"Critical error in _check_all_active_profiles: {e}")
            logging.error(f"File checker continuing despite critical error...")

    def _scan_profile_folders(self, profile_name, details):
        """Scan a profile's folders for files to process"""
        try:
            folders_to_process = []
            
            # Check JPEG folder for PDFs
            try:
                jpeg_folder = details.get('JPEG', '')
                if jpeg_folder and os.path.exists(jpeg_folder):
                    folders_to_process.extend(self._find_folders_with_new_files(jpeg_folder, '.pdf'))
            except Exception as e:
                logging.error(f"Error scanning JPEG folder for profile {profile_name}: {e}")
            
            # Check TIFF folder for PDFs and JPEGs
            try:
                tiff_folder = details.get('TIFF', '')
                if tiff_folder and os.path.exists(tiff_folder):
                    folders_to_process.extend(self._find_folders_with_new_files(tiff_folder, '.pdf'))
                    folders_to_process.extend(self._find_folders_with_new_files(tiff_folder, '.jpg'))
                    folders_to_process.extend(self._find_folders_with_new_files(tiff_folder, '.jpeg'))
            except Exception as e:
                logging.error(f"Error scanning TIFF folder for profile {profile_name}: {e}")
            
            if folders_to_process:
                logging.info(f"Profile {profile_name} found {len(folders_to_process)} folders with new files to process")
                
                # Trigger processing by touching each folder
                for folder_path in folders_to_process:
                    try:
                        self._trigger_folder_processing(folder_path)
                    except Exception as e:
                        logging.error(f"Error triggering processing for folder {folder_path}: {e}")
                        continue
                    
        except Exception as e:
            logging.error(f"Critical error scanning folders for profile {profile_name}: {e}")
            logging.error(f"Continuing with file checker despite error...")

    def _find_folders_with_new_files(self, base_folder, file_extension):
        """Recursively find folders containing NEW or MODIFIED files with the specified extension"""
        folders_with_new_files = []
        try:
            for root, dirs, files in os.walk(base_folder):
                try:
                    # Check if this directory contains new or modified files with the target extension
                    has_new_files = False
                    
                    for file in files:
                        if file.lower().endswith(file_extension.lower()):
                            file_path = os.path.join(root, file)
                            if self._is_file_new_or_modified(file_path):
                                has_new_files = True
                                break
                    
                    if has_new_files:
                        folders_with_new_files.append(root)
                        
                except (OSError, PermissionError):
                    continue  # Skip folders we can't read
        except (OSError, PermissionError):
            pass  # Skip if we can't read the base folder
        
        return folders_with_new_files
    
    def _is_file_new_or_modified(self, file_path):
        """Check if a file is new or has been modified since last processing"""
        try:
            # Get file modification time
            file_mtime = os.path.getmtime(file_path)
            current_time = time.time()
            
            with self._file_lock:
                # Check if we've seen this file before
                if file_path in self._processed_files:
                    last_mtime, last_process_time = self._processed_files[file_path]
                    
                    # If file hasn't been modified since we last processed it, skip it
                    if file_mtime <= last_mtime:
                        return False
                    
                    # If we processed it very recently (within 30 seconds), skip it
                    if current_time - last_process_time < 30:
                        return False
                
                # Mark file as being processed now
                self._processed_files[file_path] = (file_mtime, current_time)
                
                # Clean up old entries (keep only last 1000 files to prevent memory bloat)
                if len(self._processed_files) > 1000:
                    # Remove oldest 200 entries
                    sorted_files = sorted(self._processed_files.items(), key=lambda x: x[1][1])
                    for old_file, _ in sorted_files[:200]:
                        del self._processed_files[old_file]
                
                return True
                
        except (OSError, PermissionError) as e:
            logging.error(f"Error checking file {file_path}: {e}")
            return False

    def _trigger_folder_processing(self, folder_path):
        """Trigger processing of a folder by updating its modification time"""
        try:
            # Verify folder exists before attempting to touch it
            if not os.path.exists(folder_path):
                logging.warning(f"Folder does not exist, skipping processing: {folder_path}")
                return
            
            if not os.path.isdir(folder_path):
                logging.warning(f"Path is not a directory, skipping processing: {folder_path}")
                return
            
            # Touch the folder to trigger watchdog event
            current_time = time.time()
            os.utime(folder_path, (current_time, current_time))
            logging.info(f"Triggered processing of folder with new files: {folder_path}")
        except PermissionError as e:
            logging.error(f"Permission denied when triggering processing of folder {folder_path}: {e}")
        except OSError as e:
            logging.error(f"OS error when triggering processing of folder {folder_path}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error triggering processing of folder {folder_path}: {e}")
            logging.error(f"Continuing with file checker despite folder error...")


    def _do_config_save(self):
        """Actually save the config file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info("Config saved to disk")
        except Exception as e:
            logging.error(f"Error saving config: {e}")

    def load_config(self):
        """load settings from json file"""
        try:
            if not os.path.exists(self.config_file):
                # create empty template if missing
                self.config = {
                    "network_folder": "", 
                    "profiles": {}, 
                    "core_cap": multiprocessing.cpu_count(),
                    "max_cores_per_profile": 6  # default max per profile
                }
                self.save_config()
            else:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
                    
                # add missing fields to older configs
                if "max_cores_per_profile" not in self.config:
                    self.config["max_cores_per_profile"] = 6
                    self.save_config()
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            # Create default config on error
            self.config = {
                "network_folder": "", 
                "profiles": {}, 
                "core_cap": multiprocessing.cpu_count(),
                "max_cores_per_profile": 6
            }

    def save_config(self):
        """Mark config as needing save instead of doing it immediately"""
        with self._config_lock:
            self._config_changed = True

    def force_save_config(self):
        """Force immediate config save (for critical operations)"""
        with self._config_lock:
            self._do_config_save()
            self._config_changed = False
            self._last_config_save = time.time()

    def update_core_cap(self, new_core_cap):
        """change the max cores setting"""
        self.config['core_cap'] = new_core_cap
        self.save_config()

    def update_max_cores_per_profile(self, new_max):
        """change max cores per profile setting"""
        self.config['max_cores_per_profile'] = new_max
        self.save_config()

    def get_core_cap(self):
        """get current core cap setting"""
        return self.config.get('core_cap', multiprocessing.cpu_count())

    def get_max_cores_per_profile(self):
        """get max cores per profile setting"""
        return self.config.get('max_cores_per_profile', 6)

    def update_network_folder(self, folder):
        """set the main network folder path"""
        self.config['network_folder'] = folder
        self.save_config()

    def get_profiles_with_status(self):
        """get all profiles with their status info"""
        profiles = self.config.get('profiles', {}).copy()  # Make a copy to avoid modifying original
        
        # Update display status for transitioning profiles
        for profile_name in self.transitioning_profiles:
            if profile_name in profiles:
                current_status = profiles[profile_name]['status']
                
                # Check if display_status is being stuck in a transitioning state
                current_display = profiles[profile_name].get('display_status', current_status)
                
                # Only update if in a valid transition
                if current_status == "Active" and current_display != "Active":
                    profiles[profile_name]['display_status'] = "Activating..."
                elif current_status == "Paused" and current_display != "Paused":
                    profiles[profile_name]['display_status'] = "Pausing..."
        
        # Check for stuck transitioning states - fix if actual processes don't match display state
        for profile_name, details in profiles.items():
            current_status = details['status']
            display_status = details.get('display_status', current_status)
            
            # If display says activating but it's no longer in transitioning set and processes exist
            if display_status == "Activating..." and profile_name not in self.transitioning_profiles:
                if profile_name in self.processes:
                    # Processes are running, so it's active
                    profiles[profile_name]['display_status'] = "Active"
                    
            # If display says pausing but it's no longer in transitioning set and no processes
            if display_status == "Pausing..." and profile_name not in self.transitioning_profiles:
                if profile_name not in self.processes:
                    # No processes, so it's fully paused
                    profiles[profile_name]['display_status'] = "Paused"
        
        return profiles

    def add_profile(self, profile_name):
        """add a new job profile with needed directories"""
        base_folder = os.path.join(self.config['network_folder'], profile_name)
        jpeg_folder = os.path.join(base_folder, "JPEG")
        tiff_folder = os.path.join(base_folder, "TIFF")
        complete_folder = os.path.join(base_folder, "COMPLETE")

        # create the directories
        os.makedirs(jpeg_folder, exist_ok=True)
        os.makedirs(tiff_folder, exist_ok=True)
        os.makedirs(complete_folder, exist_ok=True)

        # figure out reasonable core allocation
        profiles = self.get_profiles_with_status()
        total_allocated_cores = 0
        for existing_profile, details in profiles.items():
            cores_per_processor = details.get('cores_per_processor', 3)
            total_allocated_cores += cores_per_processor * 2  # both jpeg and tiff
        
        available_cores = max(2, self.get_core_cap() - total_allocated_cores)
        max_cores_per_profile = self.get_max_cores_per_profile()
        
        # set reasonable cores_per_processor
        # divide by 2 since each profile needs jpeg + tiff
        cores_per_processor = max(1, min(3, min(available_cores, max_cores_per_profile) // 2))
        
        logging.info(f"Adding profile {profile_name} with {cores_per_processor} cores per processor")

        # add to config
        self.config['profiles'][profile_name] = {
            "JPEG": jpeg_folder,
            "TIFF": tiff_folder,
            "COMPLETE": complete_folder,
            "status": "Active",
            "display_status": "Active",
            "cores_per_processor": cores_per_processor
        }
        self.save_config()

        # start processors when adding
        return profile_name

    def remove_profile(self, profile_name):
        """remove a job profile and its directories"""
        base_folder = os.path.join(self.config['network_folder'], profile_name)

        # stop any running processors first
        self.stop_processor(profile_name)

        # try to remove the directory
        try:
            shutil.rmtree(base_folder)
            logging.info(f"Successfully removed profile folder: {base_folder}")
        except OSError as e:
            logging.error(f"Error removing profile folder: {e}")

        # remove from config
        if profile_name in self.config['profiles']:
            del self.config['profiles'][profile_name]
            self.save_config()
            
        # rebalance cores after removing
        self.rebalance_cores()

    def rebalance_cores(self):
        """redistribute cores among active profiles"""
        try:
            profiles = self.get_profiles_with_status()
            active_profiles = {name: details for name, details in profiles.items() 
                              if details['status'] == 'Active'}
            
            if not active_profiles:
                return  # nothing to do if no active profiles
            
            # calculate optimal core allocation
            core_cap = self.get_core_cap()
            max_cores_per_profile = self.get_max_cores_per_profile()
            
            # each profile runs 2 processors (jpeg + tiff)
            optimal_cores_per_processor = min(
                max_cores_per_profile // 2,
                max(1, (core_cap // (len(active_profiles) * 2)))
            )
            
            logging.info(f"Rebalancing cores: {len(active_profiles)} active profiles, "
                        f"optimal is {optimal_cores_per_processor} cores per processor")
            
            # update each profile if needed
            for profile_name, details in active_profiles.items():
                current_cores = details.get('cores_per_processor', 3)
                if current_cores != optimal_cores_per_processor:
                    logging.info(f"Adjusting {profile_name} from {current_cores} to "
                                f"{optimal_cores_per_processor} cores per processor")
                    
                    # update in config
                    self.config['profiles'][profile_name]['cores_per_processor'] = optimal_cores_per_processor
                    
                    # restart if active
                    if profile_name in self.processes:
                        self.stop_processor(profile_name)
                        profile = self.config['profiles'][profile_name]
                        self.start_processor(profile_name, "jpeg_processor.py", 
                                            profile["JPEG"], profile["COMPLETE"], 
                                            optimal_cores_per_processor)
                        self.start_processor(profile_name, "tiff_processor.py", 
                                            profile["TIFF"], profile["COMPLETE"], 
                                            optimal_cores_per_processor)
            
            # save changes
            self.save_config()
            
        except Exception as e:
            logging.error(f"Error in rebalance_cores: {e}")
            logging.error(traceback.format_exc())

    def pause_profile(self, profile_name):
        """mark profile as paused and stop processors"""
        if profile_name in self.config['profiles']:
            self.transitioning_profiles.add(profile_name)
            
            try:
                # stop running processors
                self.stop_processor(profile_name)
                
                # update status
                self.config['profiles'][profile_name]['status'] = "Paused"
                self.config['profiles'][profile_name]['display_status'] = "Paused"
                self.save_config()
                
                logging.info(f"Profile {profile_name} paused successfully")
            except Exception as e:
                logging.error(f"Error pausing profile {profile_name}: {e}")
            finally:
                # done transitioning
                if profile_name in self.transitioning_profiles:
                    self.transitioning_profiles.remove(profile_name)
                    
            # rebalance cores after pausing
            self.rebalance_cores()

    def unpause_profile(self, profile_name):
        """mark profile as active and start processors"""
        if profile_name in self.config['profiles']:
            # Add to transitioning profiles set
            self.transitioning_profiles.add(profile_name)
            
            try:
                # Update status immediately
                self.config['profiles'][profile_name]['status'] = "Active"
                self.config['profiles'][profile_name]['display_status'] = "Activating..."
                self.save_config()
                
                # Rebalance before starting
                self.rebalance_cores()
                
                # Restart processors with updated core settings
                profile = self.config['profiles'][profile_name]
                cores_per_processor = profile.get("cores_per_processor", 3)  # default if not set
                
                # Start processors
                jpeg_success = self.start_processor(profile_name, "jpeg_processor.py", profile["JPEG"], profile["COMPLETE"], cores_per_processor)
                tiff_success = self.start_processor(profile_name, "tiff_processor.py", profile["TIFF"], profile["COMPLETE"], cores_per_processor)
                
                if jpeg_success and tiff_success:
                    # Update display status after successful start
                    self.config['profiles'][profile_name]['display_status'] = "Active"
                    logging.info(f"Profile {profile_name} started successfully")
                else:
                    # Mark as error if failed to start
                    self.config['profiles'][profile_name]['status'] = "Paused"
                    self.config['profiles'][profile_name]['display_status'] = "Error"
                    logging.error(f"Failed to start processors for profile {profile_name}")
                
                self.save_config()
                
            except Exception as e:
                logging.error(f"Error unpausing profile {profile_name}: {e}")
                logging.error(traceback.format_exc())
                # Mark as error on exception
                self.config['profiles'][profile_name]['status'] = "Paused"
                self.config['profiles'][profile_name]['display_status'] = "Error"
                self.save_config()
            finally:
                # Always clean up transitioning set
                if profile_name in self.transitioning_profiles:
                    self.transitioning_profiles.remove(profile_name)

    def toggle_profile_status(self, profile_name):
        """toggle between active and paused"""
        if profile_name in self.config['profiles']:
            if profile_name in self.transitioning_profiles:
                logging.warning(f"Profile {profile_name} is already transitioning, cannot toggle")
                return
                
            current_status = self.config['profiles'][profile_name]['status']
            if current_status == "Active":
                self.pause_profile(profile_name)
            else:
                self.unpause_profile(profile_name)

    def start_processor(self, profile_name, processor_name, watch_dir, output_dir, cores=3):
        """start a processor (jpeg/tiff) with specified cores"""
        if profile_name not in self.processes:
            self.processes[profile_name] = {}

        # create log file
        log_file = os.path.join(script_dir, f"{profile_name}_{processor_name}.log")
        
        try:
            # use explicit python interpreter
            command = [
                sys.executable,  # use current python
                os.path.join(script_dir, processor_name),  # full path to script
                f'--watch-dir={watch_dir}',
                f'--output-dir={output_dir}',
                f'--max-workers={cores}'
            ]

            # open log for writing
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log, 
                    stderr=log, 
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            
            logging.info(f"Started {processor_name} for profile {profile_name} with {cores} cores")
            logging.info(f"Command: {' '.join(command)}")
            
            self.processes[profile_name][processor_name] = process
            
            return process
        
        except Exception as e:
            logging.error(f"Error starting {processor_name} for profile {profile_name}: {e}")
            logging.error(traceback.format_exc())
            return None
            
    def stop_processor(self, profile_name):
        """Stop processors for a profile with retries for termination."""
        if profile_name in self.processes:
            for processor_name, process in self.processes[profile_name].items():
                try:
                    logging.info(f"Stopping {processor_name} for profile {profile_name}")
                    
                    process.terminate()
                    retries = 0
                    max_retries = 10  # max retries (10, 0.5 seconds in between)

                    # wait to terminate 
                    while retries < max_retries:
                        if process.poll() is not None:  # check 
                            logging.info(f"Process {processor_name} terminated successfully after {retries * 0.5} seconds")
                            break
                        time.sleep(0.5)
                        retries += 1
                    
                    # force kill if still running
                    if process.poll() is None:
                        logging.warning(f"Process {processor_name} did not terminate gracefully, force killing...")
                        if sys.platform == 'win32':
                            subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
                        else:
                            os.kill(process.pid, 9)
                        logging.info(f"Force killed {processor_name} for profile {profile_name}")
                    
                except Exception as e:
                    logging.error(f"Error terminating {processor_name} for profile {profile_name}: {e}")
                    logging.error(traceback.format_exc())
            
            # clean up
            del self.processes[profile_name]

    def start_all_profiles(self):
        """Force start ALL profiles regardless of current status"""
        # Rebalance cores first for optimal allocation
        self.rebalance_cores()
        
        # Start all profiles regardless of current status
        started_count = 0
        for profile_name, details in self.config['profiles'].items():
            try:
                # Set status to Active first
                self.config['profiles'][profile_name]['status'] = "Active"
                self.config['profiles'][profile_name]['display_status'] = "Active"
                
                # Get cores setting
                cores_per_processor = details.get("cores_per_processor", 3)  # default if not set
                
                # Stop any existing processes for this profile (in case it's already running)
                if profile_name in self.processes:
                    self.stop_processor(profile_name)
                
                # Start the processors
                jpeg_result = self.start_processor(profile_name, "jpeg_processor.py", 
                                                details["JPEG"], details["COMPLETE"], 
                                                cores_per_processor)
                tiff_result = self.start_processor(profile_name, "tiff_processor.py", 
                                                details["TIFF"], details["COMPLETE"], 
                                                cores_per_processor)
                
                if jpeg_result and tiff_result:
                    logging.info(f"Successfully started both processors for profile {profile_name}")
                    started_count += 1
                else:
                    logging.error(f"Failed to start one or more processors for profile {profile_name}")
                    # Mark as error if failed
                    self.config['profiles'][profile_name]['status'] = "Paused"
                    self.config['profiles'][profile_name]['display_status'] = "Error"
                    
            except Exception as e:
                logging.error(f"Error starting profile {profile_name}: {e}")
                self.config['profiles'][profile_name]['status'] = "Paused"
                self.config['profiles'][profile_name]['display_status'] = "Error"
        
        # Save config after updating all statuses
        self.save_config()
        
        logging.info(f"Force-started {started_count} profiles")
        return started_count

    def stop_all_profiles(self):
        """stop all running profiles"""
        stopped_count = 0
        for profile_name in list(self.processes.keys()):
            try:
                logging.info(f"Stopping profile: {profile_name}")
                self.stop_processor(profile_name)
                if profile_name in self.config['profiles']:
                    self.config['profiles'][profile_name]['status'] = "Paused"
                    self.config['profiles'][profile_name]['display_status'] = "Paused"
                stopped_count += 1
            except Exception as e:
                logging.error(f"Error stopping profile {profile_name}: {e}")
                
        self.save_config()
        logging.info(f"Stopped {stopped_count} profiles")

    def update_cores_per_profile(self, profile_name, total_cores):
        """update cores for a profile (total divided between processors)"""
        if profile_name in self.config['profiles']:
            # stay within allowed range
            max_cores = self.get_max_cores_per_profile()
            total_cores = max(2, min(total_cores, max_cores))
            
            # split between processors
            cores_per_processor = total_cores // 2
            
            # update config
            self.config['profiles'][profile_name]['cores_per_processor'] = cores_per_processor
            self.save_config()
            
            logging.info(f"Updated {profile_name} to use {cores_per_processor} cores per processor (total: {total_cores})")
            
            # restart if active
            self.update_cores_per_processor(profile_name, cores_per_processor)
            
            return True
        return False

    def update_cores_per_processor(self, profile_name, cores_per_processor):
        """update cores_per_processor setting for a profile"""
        try:
            if profile_name in self.config['profiles']:
                # enforce minimum
                cores_per_processor = max(1, cores_per_processor)
                
                # update setting
                self.config['profiles'][profile_name]['cores_per_processor'] = cores_per_processor
                
                # save config
                self.save_config()
                
                logging.info(f"Updated {profile_name} to use {cores_per_processor} cores per processor")
                
                # restart if active
                if self.config['profiles'][profile_name].get('status') == 'Active':
                    logging.info(f"Restarting {profile_name} to apply core changes")
                    self.stop_processor(profile_name)
                    
                    # get paths from profile
                    jpeg_path = self.config['profiles'][profile_name].get('JPEG', '')
                    tiff_path = self.config['profiles'][profile_name].get('TIFF', '')
                    complete_path = self.config['profiles'][profile_name].get('COMPLETE', '')
                    
                    # restart with new settings
                    self.start_processor(profile_name, "jpeg_processor.py", jpeg_path, complete_path, cores_per_processor)
                    self.start_processor(profile_name, "tiff_processor.py", tiff_path, complete_path, cores_per_processor)
                    
                return True
            return False
        except Exception as e:
            logging.error(f"Error updating cores per processor: {e}")
            logging.error(traceback.format_exc())
            return False