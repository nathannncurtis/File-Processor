import os
import sys
import json
import shutil
import subprocess
import multiprocessing
import time
import logging
import traceback

# Determine the script directory for absolute paths
if getattr(sys, 'frozen', False):
    # If running as a frozen exe
    script_dir = os.path.dirname(sys.executable)
else:
    # If running as a regular Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

# Create log file with absolute path
log_file = os.path.join(script_dir, "job_manager.log")
print(f"Job Manager using log file: {log_file}")

# Configure logging to both file and console
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add console handler to see logs in terminal too
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# Log startup information
logging.info(f"Started job_manager")
logging.info(f"Python executable: {sys.executable}")
logging.info(f"Script directory: {script_dir}")
logging.info(f"Current working directory: {os.getcwd()}")

class JobManager:
    def __init__(self, config_file):
        """Initialize the JobManager with a path to the config file."""
        self.config_file = config_file
        self.processes = {}  # Keep track of running processes (one per job)
        self.load_config()
        self.transitioning_profiles = set()  # Track profiles currently changing state
        
        # Log initialization
        logging.info(f"JobManager initialized with config file: {config_file}")
        if not os.path.exists(config_file):
            logging.warning(f"Config file does not exist, will create: {config_file}")

    def load_config(self):
        """Loads the configuration from the JSON file."""
        if not os.path.exists(self.config_file):
            # If the config file doesn't exist, create a blank template
            self.config = {
                "network_folder": "", 
                "profiles": {}, 
                "core_cap": multiprocessing.cpu_count(),
                "max_cores_per_profile": 6  # Default max cores per profile
            }
            self.save_config()
        else:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                
            # Ensure new fields exist in older config files
            if "max_cores_per_profile" not in self.config:
                self.config["max_cores_per_profile"] = 6
                self.save_config()

    def save_config(self):
        """Saves the current configuration to the JSON file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)

    def update_core_cap(self, new_core_cap):
        """Updates the core cap in the config file."""
        self.config['core_cap'] = new_core_cap
        self.save_config()

    def update_max_cores_per_profile(self, new_max):
        """Updates the maximum cores per profile setting."""
        self.config['max_cores_per_profile'] = new_max
        self.save_config()

    def get_core_cap(self):
        """Returns the current core cap from the config file."""
        return self.config.get('core_cap', multiprocessing.cpu_count())

    def get_max_cores_per_profile(self):
        """Returns the maximum cores per profile setting."""
        return self.config.get('max_cores_per_profile', 6)

    def update_network_folder(self, folder):
        """Updates the top-level network folder in the config."""
        self.config['network_folder'] = folder
        self.save_config()

    def get_profiles_with_status(self):
        """Returns a dictionary of profiles with their status (active/paused/transitioning)."""
        profiles = self.config.get('profiles', {})
        
        # Mark transitioning profiles
        for profile_name in self.transitioning_profiles:
            if profile_name in profiles:
                current_status = profiles[profile_name]['status']
                if current_status == "Active":
                    profiles[profile_name]['display_status'] = "Pausing..."
                else:
                    profiles[profile_name]['display_status'] = "Activating..."
            
        return profiles

    def add_profile(self, profile_name):
        """Adds a new job profile, creates necessary directories, and starts the processors."""
        base_folder = os.path.join(self.config['network_folder'], profile_name)
        jpeg_folder = os.path.join(base_folder, "JPEG")
        tiff_folder = os.path.join(base_folder, "TIFF")
        complete_folder = os.path.join(base_folder, "COMPLETE")

        # Create the required directories for the profile
        os.makedirs(jpeg_folder, exist_ok=True)
        os.makedirs(tiff_folder, exist_ok=True)
        os.makedirs(complete_folder, exist_ok=True)

        # Calculate available cores for this profile
        profiles = self.get_profiles_with_status()
        total_allocated_cores = 0
        for existing_profile, details in profiles.items():
            cores_per_processor = details.get('cores_per_processor', 3)
            total_allocated_cores += cores_per_processor * 2  # Total is JPEG + TIFF
        
        available_cores = max(2, self.get_core_cap() - total_allocated_cores)
        max_cores_per_profile = self.get_max_cores_per_profile()
        
        # Calculate reasonable cores_per_processor value
        # Each profile gets JPEG + TIFF processors, so divide by 2
        cores_per_processor = max(1, min(3, min(available_cores, max_cores_per_profile) // 2))
        
        logging.info(f"Adding profile {profile_name} with {cores_per_processor} cores per processor")

        # Add the profile to the configuration
        self.config['profiles'][profile_name] = {
            "JPEG": jpeg_folder,
            "TIFF": tiff_folder,
            "COMPLETE": complete_folder,
            "status": "Active",
            "display_status": "Active",
            "cores_per_processor": cores_per_processor
        }
        self.save_config()

        # Start the processors when adding a new job
        return profile_name

    def remove_profile(self, profile_name):
        """Removes a job profile and its corresponding directories."""
        base_folder = os.path.join(self.config['network_folder'], profile_name)

        # Stop any running processors for this profile before removing
        self.stop_processor(profile_name)

        # Attempt to remove the profile's directory
        try:
            shutil.rmtree(base_folder)
            logging.info(f"Successfully removed profile folder: {base_folder}")
        except OSError as e:
            logging.error(f"Error removing profile folder: {e}")

        # Remove the profile from the configuration
        if profile_name in self.config['profiles']:
            del self.config['profiles'][profile_name]
            self.save_config()
            
        # After removing a profile, rebalance cores if needed
        self.rebalance_cores()

    def rebalance_cores(self):
        """Rebalance cores across active profiles to optimize usage."""
        try:
            profiles = self.get_profiles_with_status()
            active_profiles = {name: details for name, details in profiles.items() 
                              if details['status'] == 'Active'}
            
            if not active_profiles:
                return  # No active profiles to rebalance
            
            # Calculate optimal cores per processor based on core cap
            core_cap = self.get_core_cap()
            max_cores_per_profile = self.get_max_cores_per_profile()
            
            # Each profile uses 2 processors (JPEG + TIFF)
            optimal_cores_per_processor = min(
                max_cores_per_profile // 2,
                max(1, (core_cap // (len(active_profiles) * 2)))
            )
            
            logging.info(f"Rebalancing cores: {len(active_profiles)} active profiles, "
                        f"optimal is {optimal_cores_per_processor} cores per processor")
            
            # Update each profile if needed
            for profile_name, details in active_profiles.items():
                current_cores = details.get('cores_per_processor', 3)
                if current_cores != optimal_cores_per_processor:
                    logging.info(f"Adjusting {profile_name} from {current_cores} to "
                                f"{optimal_cores_per_processor} cores per processor")
                    
                    # Update in config
                    self.config['profiles'][profile_name]['cores_per_processor'] = optimal_cores_per_processor
                    
                    # Restart if active
                    if profile_name in self.processes:
                        self.stop_processor(profile_name)
                        profile = self.config['profiles'][profile_name]
                        self.start_processor(profile_name, "jpeg_processor.py", 
                                            profile["JPEG"], profile["COMPLETE"], 
                                            optimal_cores_per_processor)
                        self.start_processor(profile_name, "tiff_processor.py", 
                                            profile["TIFF"], profile["COMPLETE"], 
                                            optimal_cores_per_processor)
            
            # Save the updated config
            self.save_config()
            
        except Exception as e:
            logging.error(f"Error in rebalance_cores: {e}")
            logging.error(traceback.format_exc())

    def pause_profile(self, profile_name):
        """Marks a profile as paused in the configuration and stops the processors."""
        if profile_name in self.config['profiles']:
            self.transitioning_profiles.add(profile_name)
            
            # Stop the processors
            self.stop_processor(profile_name)
            
            # Update the status
            self.config['profiles'][profile_name]['status'] = "Paused"
            self.config['profiles'][profile_name]['display_status'] = "Paused"
            self.save_config()
            
            # Remove from transitioning set
            if profile_name in self.transitioning_profiles:
                self.transitioning_profiles.remove(profile_name)
                
            # After pausing a profile, rebalance cores if needed
            self.rebalance_cores()

    def unpause_profile(self, profile_name):
        """Marks a profile as active in the configuration and restarts the processors."""
        if profile_name in self.config['profiles']:
            self.transitioning_profiles.add(profile_name)
            
            # Update the status immediately
            self.config['profiles'][profile_name]['status'] = "Active"
            self.config['profiles'][profile_name]['display_status'] = "Active"
            self.save_config()
            
            # Rebalance cores before starting
            self.rebalance_cores()
            
            # Restart the processors with potentially updated core settings
            profile = self.config['profiles'][profile_name]
            cores_per_processor = profile.get("cores_per_processor", 3)  # Default if not specified
            
            self.start_processor(profile_name, "jpeg_processor.py", profile["JPEG"], profile["COMPLETE"], cores_per_processor)
            self.start_processor(profile_name, "tiff_processor.py", profile["TIFF"], profile["COMPLETE"], cores_per_processor)
            
            # Remove from transitioning set
            if profile_name in self.transitioning_profiles:
                self.transitioning_profiles.remove(profile_name)

    def toggle_profile_status(self, profile_name):
        """Toggles the profile status between active and paused."""
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
        """Start a processor (JPEG/TIFF) using the appropriate .exe or .py file with specified cores."""
        if profile_name not in self.processes:
            self.processes[profile_name] = {}

        # Create log file in the script directory
        log_file = os.path.join(script_dir, f"{profile_name}_{processor_name}.log")
        
        try:
            # Use sys.executable to explicitly use the Python interpreter
            command = [
                sys.executable,  # Explicitly use Python interpreter
                os.path.join(script_dir, processor_name),  # Full path to the script
                f'--watch-dir={watch_dir}',
                f'--output-dir={output_dir}',
                f'--max-workers={cores}'
            ]

            # Open log file for writing
            with open(log_file, "w") as log:
                process = subprocess.Popen(
                    command,
                    stdout=log, 
                    stderr=log, 
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            
            logging.info(f"Started {processor_name} for profile {profile_name}")
            logging.info(f"Command: {' '.join(command)}")
            
            self.processes[profile_name][processor_name] = process
            
            return process
        
        except Exception as e:
            logging.error(f"Error starting {processor_name} for profile {profile_name}: {e}")
            logging.error(traceback.format_exc())
            return None
            
    def stop_processor(self, profile_name):
        """Stop any running processors for the given profile."""
        if profile_name in self.processes:
            for processor_name, process in self.processes[profile_name].items():
                try:
                    logging.info(f"Stopping {processor_name} for profile {profile_name}")
                    
                    process.terminate()
                    # Wait for up to 5 seconds for the process to terminate
                    for i in range(10):
                        if process.poll() is not None:
                            logging.info(f"Process terminated normally after {(i+1)*0.5} seconds")
                            break
                        time.sleep(0.5)
                    
                    # If it's still running, force kill
                    if process.poll() is None:
                        logging.info(f"Process did not terminate gracefully, force killing...")
                        if sys.platform == 'win32':
                            subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
                        else:
                            os.kill(process.pid, 9)
                            
                    logging.info(f"Terminated {processor_name} for profile {profile_name}")
                except Exception as e:
                    logging.error(f"Error terminating {processor_name} for profile {profile_name}: {e}")
                    logging.error(traceback.format_exc())
            
            # Clean up processes dictionary
            del self.processes[profile_name]

    def start_all_profiles(self):
        """Start all profiles marked as Active in the configuration."""
        # First, rebalance cores to ensure optimal allocation
        self.rebalance_cores()
        
        # Now start all active profiles
        active_count = 0
        for profile_name, details in self.config['profiles'].items():
            if details['status'] == "Active":
                cores_per_processor = details.get("cores_per_processor", 3)  # Default if not specified
                jpeg_result = self.start_processor(profile_name, "jpeg_processor.py", details["JPEG"], details["COMPLETE"], cores_per_processor)
                tiff_result = self.start_processor(profile_name, "tiff_processor.py", details["TIFF"], details["COMPLETE"], cores_per_processor)
                
                if jpeg_result and tiff_result:
                    logging.info(f"Successfully started both processors for profile {profile_name}")
                    active_count += 1
                else:
                    logging.error(f"Failed to start one or more processors for profile {profile_name}")
        
        logging.info(f"Started {active_count} active profiles")

    def stop_all_profiles(self):
        """Stop all running profiles."""
        stopped_count = 0
        for profile_name in list(self.processes.keys()):
            logging.info(f"Stopping profile: {profile_name}")
            self.stop_processor(profile_name)
            if profile_name in self.config['profiles']:
                self.config['profiles'][profile_name]['status'] = "Paused"
                self.config['profiles'][profile_name]['display_status'] = "Paused"
            stopped_count += 1
            
        self.save_config()
        logging.info(f"Stopped {stopped_count} profiles")

    def update_cores_per_profile(self, profile_name, total_cores):
        """Update the number of cores allocated to a profile (total cores divided between processors)."""
        if profile_name in self.config['profiles']:
            # Ensure within allowed range
            max_cores = self.get_max_cores_per_profile()
            total_cores = max(2, min(total_cores, max_cores))
            
            # Calculate cores per processor (total divided by 2)
            cores_per_processor = total_cores // 2
            
            # Update in config
            self.config['profiles'][profile_name]['cores_per_processor'] = cores_per_processor
            self.save_config()
            
            logging.info(f"Updated {profile_name} to use {cores_per_processor} cores per processor (total: {total_cores})")
            
            # If active, restart the profile with new settings
            self.update_cores_per_processor(profile_name, cores_per_processor)
            
            return True
        return False

    def update_cores_per_processor(self, profile_name, cores_per_processor):
        """Update the cores_per_processor setting for a specific profile."""
        try:
            if profile_name in self.config['profiles']:
                # Ensure cores_per_processor is at least 1
                cores_per_processor = max(1, cores_per_processor)
                
                # Update the cores_per_processor setting in the profile
                self.config['profiles'][profile_name]['cores_per_processor'] = cores_per_processor
                
                # Save the configuration
                self.save_config()
                
                logging.info(f"Updated {profile_name} to use {cores_per_processor} cores per processor")
                
                # Restart the profile if it's active
                if self.config['profiles'][profile_name].get('status') == 'Active':
                    logging.info(f"Restarting {profile_name} to apply core changes")
                    self.stop_processor(profile_name)
                    
                    # Get paths from profile
                    jpeg_path = self.config['profiles'][profile_name].get('JPEG', '')
                    tiff_path = self.config['profiles'][profile_name].get('TIFF', '')
                    complete_path = self.config['profiles'][profile_name].get('COMPLETE', '')
                    
                    # Restart with new core settings
                    self.start_processor(profile_name, "jpeg_processor.py", jpeg_path, complete_path, cores_per_processor)
                    self.start_processor(profile_name, "tiff_processor.py", tiff_path, complete_path, cores_per_processor)
                    
                return True
            return False
        except Exception as e:
            logging.error(f"Error updating cores per processor: {e}")
            logging.error(traceback.format_exc())
            return False