import multiprocessing
import subprocess
import os
import sys
import threading
import traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QListWidget, QVBoxLayout, QWidget, QLabel, QInputDialog, QSpacerItem, QSizePolicy, QSystemTrayIcon, QMenu, QAction, qApp, QDialog, QLineEdit, QSpinBox, QDialogButtonBox, QMenuBar, QMessageBox, QDesktopWidget, QHBoxLayout, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt5.QtGui import QFont, QIcon, QCursor
from time import sleep
import time

# try and import psutil for core detection
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("psutil not available. Physical core detection will be limited.")

# defer job_manager import to handle path issues
JobManager = None

# set base dir for both frozen and non-frozen modes
BASE_DIR = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, 'frozen', False) else os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, 'config.json')

print(f"Base directory: {BASE_DIR}")
print(f"Config file path: {CONFIG_FILE}")

# add the base dir to sys.path to ensure imports work
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# import job_manager
try:
    from job_manager import JobManager
except ImportError:
    print(f"Failed to import JobManager. Paths: {sys.path}")
    print(f"Files in directory: {os.listdir(BASE_DIR)}")
    print(traceback.format_exc())
    # error handled in main_ui class

class JobQueueManager(QThread):
    job_finished_signal = pyqtSignal()
    status_update_signal = pyqtSignal()
    core_rebalanced_signal = pyqtSignal()

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.active_processes = []
        self.should_stop = False
        self.last_core_check = time.time()
        self.cleanup_counter = 0  # For periodic cleanup

    def run(self):
        """Main thread loop with better performance"""
        print("JobQueueManager thread started")
        while not self.should_stop:
            try:
                # Check if any profiles need to be started
                self.start_active_profiles()
                
                # Periodic cleanup every 30 cycles (30 seconds)
                self.cleanup_counter += 1
                if self.cleanup_counter >= 30:
                    self.cleanup_dead_processes()
                    self.cleanup_counter = 0
                
                # Core rebalancing check (every 10 seconds)
                current_time = time.time()
                if current_time - self.last_core_check > 10:
                    self.check_core_balance()
                    self.last_core_check = current_time
                
                # Only emit update signal if we actually need UI updates
                # This will be much less frequent now
                
            except Exception as e:
                print(f"Error in JobQueueManager run: {e}")
            
            # Longer sleep interval since we're not driving UI updates
            time.sleep(1)

    def cleanup_dead_processes(self):
        """Remove references to dead processes"""
        try:
            profiles_to_clean = []
            
            for profile_name, processes in list(self.manager.processes.items()):
                dead_processes = []
                
                for proc_name, process in list(processes.items()):
                    try:
                        # Check if process is still alive
                        if process.poll() is not None:  # Process is dead
                            dead_processes.append(proc_name)
                            print(f"Found dead process: {profile_name}/{proc_name}")
                    except Exception as e:
                        print(f"Error checking process {profile_name}/{proc_name}: {e}")
                        dead_processes.append(proc_name)
                
                # Remove dead processes
                for proc_name in dead_processes:
                    try:
                        del processes[proc_name]
                        print(f"Cleaned up dead process: {profile_name}/{proc_name}")
                    except KeyError:
                        pass
                
                # If no processes left for this profile, mark for cleanup
                if not processes:
                    profiles_to_clean.append(profile_name)
            
            # Clean up empty profile entries
            for profile_name in profiles_to_clean:
                try:
                    del self.manager.processes[profile_name]
                    print(f"Cleaned up empty profile processes: {profile_name}")
                    
                    # Update profile status if it claims to be active but has no processes
                    if profile_name in self.manager.config['profiles']:
                        profile_config = self.manager.config['profiles'][profile_name]
                        if profile_config.get('status') == 'Active':
                            print(f"Profile {profile_name} claims Active but has no processes, marking as Error")
                            profile_config['status'] = 'Paused'
                            profile_config['display_status'] = 'Error'
                            # Signal that config changed
                            self.manager._config_changed = True
                            
                except KeyError:
                    pass
                    
        except Exception as e:
            print(f"Error in cleanup_dead_processes: {e}")

    def check_core_balance(self):
        """Check if core rebalancing is needed"""
        try:
            profiles = self.manager.get_profiles_with_status()
            
            total_allocated_cores = 0
            for profile_name, details in profiles.items():
                if details['status'] == 'Active':
                    cores_per_processor = details.get('cores_per_processor', 3)
                    total_allocated_cores += cores_per_processor * 2
            
            if total_allocated_cores > self.manager.get_core_cap():
                print(f"Core imbalance detected: {total_allocated_cores} allocated vs {self.manager.get_core_cap()} cap")
                self.manager.rebalance_cores()
                self.core_rebalanced_signal.emit()
        except Exception as e:
            print(f"Error in check_core_balance: {e}")

    def start_active_profiles(self):
        """Start processors for any active profiles that aren't already running"""
        try:
            profiles = self.manager.get_profiles_with_status()
            for profile_name, details in profiles.items():
                if details['status'] == "Active" and profile_name not in self.manager.processes:
                    print(f"Queue manager starting processors for {profile_name}")
                    cores_per_processor = details.get("cores_per_processor", 3)
                    self.manager.start_processor(profile_name, "jpeg_processor.py", details["JPEG"], details["COMPLETE"], cores_per_processor)
                    self.manager.start_processor(profile_name, "tiff_processor.py", details["TIFF"], details["COMPLETE"], cores_per_processor)
        except Exception as e:
            print(f"Error starting active profiles in JobQueueManager: {e}")

    def stop(self):
        """Stop the thread"""
        self.should_stop = True
        self.wait()

class CoreCapDialog(QDialog):
    def __init__(self, current_core_cap, physical_cores, logical_cores, current_max_per_profile, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Core Cap")
        
        layout = QVBoxLayout(self)
        
        # add sys info with both physical and logical cores
        system_info = QLabel(f"System has {physical_cores} physical cores ({logical_cores} logical processors)", self)
        system_info.setStyleSheet("color: black; font-weight: bold;")
        layout.addWidget(system_info)

        instruction_label = QLabel("Enter total number of cores to use across all profiles:", self)
        instruction_label.setStyleSheet("color: black;")
        layout.addWidget(instruction_label)

        self.spin_box = QSpinBox(self)
        self.spin_box.setRange(2, physical_cores)
        self.spin_box.setValue(min(current_core_cap, physical_cores))
        layout.addWidget(self.spin_box)

        max_per_profile_label = QLabel("Maximum cores per profile:", self)
        max_per_profile_label.setStyleSheet("color: black;")
        layout.addWidget(max_per_profile_label)

        # limit max per profile to no more than physical cores
        self.max_per_profile_spin = QSpinBox(self)
        self.max_per_profile_spin.setRange(2, min(physical_cores, 12))
        self.max_per_profile_spin.setValue(min(current_max_per_profile, physical_cores))
        layout.addWidget(self.max_per_profile_spin)
        
        # add note about per-processor allocation
        note_label = QLabel("Note: Cores are divided between JPEG and TIFF processors (each gets half)", self)
        note_label.setStyleSheet("color: black; font-style: italic; font-size: 11px;")
        layout.addWidget(note_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.center_on_cursor()

    def center_on_cursor(self):
        try:
            screen = QApplication.screenAt(QCursor.pos())
            if screen:
                screen_geometry = screen.geometry()
                self.move(screen_geometry.center() - self.rect().center())
        except Exception as e:
            print(f"Error centering dialog: {e}")

    def get_values(self):
        return {
            'core_cap': self.spin_box.value(),
            'max_per_profile': self.max_per_profile_spin.value()
        }

class ProfileNameDialog(QDialog):
    def __init__(self, max_cores_per_profile, physical_cores=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add New Job Profile")
        
        if physical_cores is None:
            physical_cores = multiprocessing.cpu_count()

        layout = QVBoxLayout(self)

        instruction_label = QLabel("Enter Profile Name:", self)
        instruction_label.setStyleSheet("color: black;")
        layout.addWidget(instruction_label)

        self.profile_input = QLineEdit(self)
        layout.addWidget(self.profile_input)

        # get max cores allowed per processor (half of max cores per profile)
        max_per_processor = max(1, max_cores_per_profile // 2)
        
        cores_label = QLabel(f"Cores per processor type (JPEG/TIFF) - Available: {max_per_processor} per processor:", self)
        cores_label.setStyleSheet("color: black;")
        layout.addWidget(cores_label)

        self.cores_spin = QSpinBox(self)
        self.cores_spin.setRange(1, max_per_processor)  # update range based on available cores
        default_cores = min(3, max_per_processor)  # default to 3 or less if fewer cores available
        self.cores_spin.setValue(default_cores)
        layout.addWidget(self.cores_spin)

        # add informational label with physical core info
        info_label = QLabel(f"This profile will use a total of {default_cores * 2} cores ({default_cores} for JPEG + {default_cores} for TIFF)", self)
        info_label.setStyleSheet("color: black; font-size: 11px;")
        layout.addWidget(info_label)
        
        # update info label when cores value changes
        self.cores_spin.valueChanged.connect(lambda value: info_label.setText(
            f"This profile will use a total of {value * 2} cores ({value} for JPEG + {value} for TIFF)"
        ))

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        self.center_on_cursor()
    def center_on_cursor(self):
        try:
            screen = QApplication.screenAt(QCursor.pos())
            if screen:
                screen_geometry = screen.geometry()
                self.move(screen_geometry.center() - self.rect().center())
        except Exception as e:
            print(f"Error centering dialog: {e}")

    def get_values(self):
        return {
            'profile_name': self.profile_input.text(),
            'cores_per_processor': self.cores_spin.value()
        }
    
class MainUI(QMainWindow):
    def __init__(self):
        super().__init__()
        print("Initializing MainUI")
        
        # check if JobManager was imported successfully
        if JobManager is None:
            QMessageBox.critical(self, "Import Error", 
                             "Failed to import JobManager. Please check the path and file.")
            qApp.quit()
            return
            
        self.setWindowTitle("File Processor")
        self.setGeometry(100, 100, 700, 500)
        
        # try to set window icon
        try:
            icon_path = os.path.join(BASE_DIR, 'processor.ico')
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                print(f"Icon loaded from {icon_path}")
            else:
                print(f"Icon not found at {icon_path}")
        except Exception as e:
            print(f"Error setting window icon: {e}")

        try:
            # get CPU count (logical processors)
            self.logical_core_count = multiprocessing.cpu_count()
            # try to get physical core count
            self.physical_core_count = self.detect_physical_cores()
            # start with physical cores as default, fall back to logical if detection fails
            self.core_count = self.physical_core_count if self.physical_core_count else self.logical_core_count
            
            print(f"Detected {self.logical_core_count} logical processors (threads)")
            print(f"Detected {self.physical_core_count} physical cores")
            print(f"Using {self.core_count} cores for constraints")
            
            # init job manager
            self.manager = JobManager(CONFIG_FILE)
            self.core_cap = min(self.manager.get_core_cap(), self.core_count)  # ensure it doesn't exceed detected cores
            self.max_cores_per_profile = min(self.manager.get_max_cores_per_profile(), self.core_count)
            self.network_folder = self.manager.config.get('network_folder', '')
        except Exception as e:
            print(f"Error initializing manager: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Initialization Error", 
                             f"Failed to initialize job manager: {str(e)}")
            qApp.quit()
            return

        # set up the ui in a try-except block
        try:
            self.init_tray()
            self.init_ui()
            self.init_menu()
        except Exception as e:
            print(f"Error initializing UI: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "UI Error", 
                             f"Failed to initialize UI: {str(e)}")
            qApp.quit()
            return

        # start the queue manager in a try-except block
        try:
            # start the queue manager to monitor and manage jobs
            self.queue_manager = JobQueueManager(self.manager)
            self.queue_manager.status_update_signal.connect(self.update_profile_list)
            self.queue_manager.core_rebalanced_signal.connect(self.handle_core_rebalance)
            self.queue_manager.start()
            
            # REMOVED: Old aggressive 1-second timer
            # Instead, use event-driven updates triggered by user actions
            
            # Add a much slower timer just for basic health checks (every 30 seconds)
            self.health_timer = QTimer(self)
            self.health_timer.timeout.connect(self.health_check)
            self.health_timer.start(30000)  # 30 seconds
            
            self.center_on_cursor()
            
            # initialize active profiles from config on startup
            # use a one-shot timer to ensure UI is fully initialized
            QTimer.singleShot(1000, self.initialize_active_profiles)
        except Exception as e:
            print(f"Error initializing threads: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Thread Error", 
                            f"Failed to initialize threads: {str(e)}")

    def health_check(self):
        """Periodic health check - much less frequent than before"""
        try:
            # Just basic sanity checks, no heavy UI updates
            profiles = self.manager.get_profiles_with_status()
            
            # Check for any profiles stuck in transitioning state
            stuck_profiles = []
            for profile_name in list(self.manager.transitioning_profiles):
                # Remove profiles stuck in transition for more than 30 seconds
                stuck_profiles.append(profile_name)
            
            for profile_name in stuck_profiles:
                self.manager.transitioning_profiles.discard(profile_name)
                print(f"Removed stuck transitioning profile: {profile_name}")
            
            # Only trigger UI update if we found stuck profiles that need attention
            if stuck_profiles:
                self.update_profile_list()
                self.update_profile_status_menu()
                
        except Exception as e:
            print(f"Error in health_check: {e}")
            
    def handle_core_rebalance(self):
        """handle the signal from JobQueueManager when cores are rebalanced"""
        print("Core rebalance detected, updating UI")
        self.update_profile_list()
        self.update_profile_status_menu()

    def detect_physical_cores(self):
        """attempt to detect physical cores (not just logical processors)"""
        try:
            # try using psutil if available (more accurate count)
            if HAVE_PSUTIL:
                physical_cores = psutil.cpu_count(logical=False)
                print(f"psutil detected {physical_cores} physical cores")
                return physical_cores
            
            # default if psutil not available - use half the logical cores as an estimate
            estimated_cores = max(1, multiprocessing.cpu_count() // 2)
            print(f"Estimated {estimated_cores} physical cores (half of logical)")
            return estimated_cores
            
        except Exception as e:
            print(f"Error detecting physical cores: {e}")
            # conservative default - use half the logical processors or at least 2
            return max(2, multiprocessing.cpu_count() // 2)

    def initialize_active_profiles(self):
        """start profiles marked as Active in the config file, after UI is initialized"""
        try:
            print("Initializing active profiles from config file")
            profiles = self.manager.get_profiles_with_status()
            for profile_name, details in profiles.items():
                print(f"Profile: {profile_name}, Status: {details.get('status', 'Unknown')}")
                
            # Call directly instead of using thread
            self.start_all_active_profiles()
            
        except Exception as e:
            print(f"Error initializing active profiles: {e}")
            print(traceback.format_exc())

    def start_all_active_profiles(self):
        """called to start all active profiles"""
        try:
            print("Starting all active profiles")
            self.manager.start_all_profiles()
            print("All active profiles should now be started")
            
            # Update UI directly since we're on main thread
            self.update_profile_list()
            self.update_profile_status_menu()
            
        except Exception as e:
            print(f"Error in start_all_active_profiles: {e}")
            print(traceback.format_exc())

    def init_tray(self):
        print("Initializing system tray")
        try:
            icon_path = os.path.join(BASE_DIR, "processor.ico")
            if os.path.exists(icon_path):
                self.tray_icon = QSystemTrayIcon(QIcon(icon_path), self)
            else:
                print(f"Warning: Icon file not found at {icon_path}")
                # create a default tray icon
                self.tray_icon = QSystemTrayIcon(self)
                
            self.tray_icon.setToolTip("File Processor App")

            self.tray_menu = QMenu(self)
            self._last_menu_items = []  # Track menu state to prevent flashing
            self.update_profile_status_menu()

            self.tray_icon.setContextMenu(self.tray_menu)
            self.tray_icon.show()
        except Exception as e:
            print(f"Error in init_tray: {e}")
            print(traceback.format_exc())
            raise

    def update_profile_status_menu(self):
        """Fixed system tray menu - only update when needed, prevent flashing"""
        try:
            # Generate new menu content first
            new_menu_items = []
            profiles_with_status = self.manager.get_profiles_with_status()
            
            for profile, details in profiles_with_status.items():
                display_status = details.get('display_status', details['status'])
                cores_per_processor = details.get('cores_per_processor', 3)
                total_cores = cores_per_processor * 2
                menu_text = f"{profile} - {display_status} - {total_cores} cores"
                new_menu_items.append(menu_text)
            
            # Only rebuild menu if content actually changed
            if hasattr(self, '_last_menu_items') and new_menu_items == self._last_menu_items:
                return  # No change, skip expensive menu rebuild
            
            self._last_menu_items = new_menu_items[:]
            
            # Clear and rebuild menu
            self.tray_menu.clear()
            
            # Add profile items
            for item_text in new_menu_items:
                profile_action = QAction(item_text, self)
                self.tray_menu.addAction(profile_action)
            
            # Add separator and standard actions
            self.tray_menu.addSeparator()
            show_action = QAction("Show", self)
            quit_action = QAction("Quit", self)
            show_action.triggered.connect(self.show)
            quit_action.triggered.connect(self.confirm_quit)
            self.tray_menu.addAction(show_action)
            self.tray_menu.addAction(quit_action)
            
        except Exception as e:
            print(f"Error updating profile status menu: {e}")
            print(traceback.format_exc())

    def init_ui(self):
        print("Initializing UI")
        
        # set stylesheet
        try:
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #2e2e2e;
                }
                QLabel {
                    color: #ffffff;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #4a90e2;
                    color: white;
                    font-size: 14px;
                    padding: 8px;
                    border: none;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #357ABD;
                }
                QListWidget {
                    background-color: #3e3e3e;
                    color: white;
                    padding: 8px;
                    font-size: 13px;
                    border: 1px solid #4a90e2;
                    border-radius: 4px;
                }
                QPushButton#start_all {
                    background-color: #4CAF50;
                }
                QPushButton#start_all:hover {
                    background-color: #45a049;
                }
                QPushButton#stop_all {
                    background-color: #f44336;
                }
                QPushButton#stop_all:hover {
                    background-color: #d32f2f;
                }
            """)
        except Exception as e:
            print(f"Error setting stylesheet: {e}")

        layout = QVBoxLayout()

        folder_label_font = QFont()
        folder_label_font.setPointSize(12)
        folder_label_font.setBold(True)

        self.folder_label = QLabel(f"Network Folder: {self.network_folder}", self)
        self.folder_label.setFont(folder_label_font)
        layout.addWidget(self.folder_label)

        set_folder_btn = QPushButton("Set Network Folder", self)
        layout.addWidget(set_folder_btn)
        set_folder_btn.clicked.connect(self.set_network_folder)

        layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # add Start All and Stop All buttons in a horizontal layout
        start_stop_layout = QHBoxLayout()
        
        start_all_btn = QPushButton("Start All Profiles", self)
        start_all_btn.setObjectName("start_all")
        start_all_btn.clicked.connect(self.start_all_profiles)
        start_stop_layout.addWidget(start_all_btn)
        
        stop_all_btn = QPushButton("Stop All Profiles", self)
        stop_all_btn.setObjectName("stop_all")
        stop_all_btn.clicked.connect(self.stop_all_profiles)
        start_stop_layout.addWidget(stop_all_btn)
        
        layout.addLayout(start_stop_layout)

        job_profile_label = QLabel("Job Profiles:", self)
        job_profile_label.setFont(folder_label_font)
        layout.addWidget(job_profile_label)

        self.job_list = QListWidget(self)
        self.job_list.setSelectionMode(QListWidget.SingleSelection)
        
        # context menu
        self.job_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.job_list.customContextMenuRequested.connect(self.show_profile_context_menu)
        
        layout.addWidget(self.job_list)

        # add and remove job profile buttons
        profile_btn_layout = QHBoxLayout()
        
        add_job_btn = QPushButton("Add Job Profile", self)
        add_job_btn.clicked.connect(self.add_job)
        profile_btn_layout.addWidget(add_job_btn)
        
        remove_job_btn = QPushButton("Remove Job Profile", self)
        remove_job_btn.clicked.connect(self.remove_job)
        profile_btn_layout.addWidget(remove_job_btn)
        
        layout.addLayout(profile_btn_layout)

        # core controls
        core_layout = QHBoxLayout()
        
        self.core_cap_btn = QPushButton(f"Set Core Limits (Total: {self.core_cap}, Max/Profile: {self.max_cores_per_profile}, Physical Cores: {self.core_count})", self)
        self.core_cap_btn.clicked.connect(self.set_core_cap)
        core_layout.addWidget(self.core_cap_btn)
        
        layout.addLayout(core_layout)

        layout.addSpacerItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Fixed))

        # single job toggle button with disabled state initially
        self.toggle_status_btn = QPushButton("Pause/Unpause Selected Job", self)
        self.toggle_status_btn.setEnabled(False)  # disabled until a profile is selected
        self.toggle_status_btn.clicked.connect(self.toggle_job_status)
        layout.addWidget(self.toggle_status_btn)

        # set the widget BEFORE connecting the itemSelectionChanged signal
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)
        
        # NOW connect the itemSelectionChanged signal after toggle_status_btn is created
        self.job_list.itemSelectionChanged.connect(self.update_toggle_button_state)
        
        # load profiles after all UI elements are created
        self.load_profiles()

    def show_profile_context_menu(self, position):
        """show a context menu when right-clicking on a profile in the list"""
        # get the item at the position
        item = self.job_list.itemAt(position)
        if not item:
            return  # no item at this position
            
        # select the item that was right-clicked
        item.setSelected(True)
            
        # extract profile name from the list item text
        profile_text = item.text()
        profile_name = profile_text.split(" - ")[0]
        status = profile_text.split(" - ")[1].lower()
        
        # create context menu
        context_menu = QMenu(self)
        
        # add actions based on current status
        if status == "active":
            pause_action = QAction("Pause Profile", self)
            pause_action.triggered.connect(lambda: self.direct_pause_profile(profile_name))
            context_menu.addAction(pause_action)
        else:
            start_action = QAction("Start Profile", self)
            start_action.triggered.connect(lambda: self.direct_start_profile(profile_name))
            context_menu.addAction(start_action)
            
        # add divider
        context_menu.addSeparator()
        
        # add remove action
        remove_action = QAction("Remove Profile", self)
        remove_action.triggered.connect(lambda: self.remove_specific_profile(profile_name))
        context_menu.addAction(remove_action)
        
        # show context menu at global position
        context_menu.exec_(self.job_list.mapToGlobal(position))
        
        # update toggle button state after action
        self.update_toggle_button_state()

    def update_toggle_button_state(self):
        """enable or disable the toggle button based on whether a profile is selected"""
        selected_items = self.job_list.selectedItems()
        self.toggle_status_btn.setEnabled(len(selected_items) > 0)
        
        # update button text based on selected profile status if any
        if selected_items:
            profile_text = selected_items[0].text()
            status = profile_text.split(" - ")[1].lower()
            
            if status == "active":
                self.toggle_status_btn.setText("Pause Selected Job")
            else:
                self.toggle_status_btn.setText("Start Selected Job")
        else:
            self.toggle_status_btn.setText("Pause/Unpause Selected Job")

    def direct_pause_profile(self, profile_name):
        """directly pause a profile and update UI"""
        try:
            print(f"Direct pause profile: {profile_name}")
            self.manager.pause_profile(profile_name)
            self.update_profile_list()
            self.update_profile_status_menu()
        except Exception as e:
            print(f"Error in direct_pause_profile: {e}")
            QMessageBox.critical(self, "Error", f"Failed to pause profile: {str(e)}")

    def direct_start_profile(self, profile_name):
        """directly start a profile and update UI"""
        try:
            print(f"Direct start profile: {profile_name}")
            self.manager.unpause_profile(profile_name)
            self.update_profile_list()
            self.update_profile_status_menu()
        except Exception as e:
            print(f"Error in direct_start_profile: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start profile: {str(e)}")

    def remove_specific_profile(self, profile_name):
        """remove a specific profile and update UI"""
        try:
            # show confirmation dialog
            reply_box = QMessageBox(self)
            reply_box.setWindowTitle("Confirm Removal")
            reply_box.setText(f"Are you sure you want to remove the profile '{profile_name}'?")
            reply_box.setStyleSheet("QLabel { color: black; }")
            reply_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.center_on_cursor(reply_box)
            reply = reply_box.exec_()
            
            if reply == QMessageBox.Yes:
                # make sure to stop the profile before removing it
                try:
                    if profile_name in self.manager.processes:
                        print(f"Stopping profile {profile_name} before removal")
                        self.manager.stop_processor(profile_name)
                except Exception as stop_error:
                    print(f"Error stopping profile before removal: {stop_error}")
                
                # now attempt to remove the profile
                try:
                    print(f"Removing profile {profile_name}")
                    self.manager.remove_profile(profile_name)
                    print(f"Successfully removed profile {profile_name}")
                    
                    # update UI after removal
                    self.load_profiles()
                    self.update_profile_status_menu()
                    
                except Exception as remove_error:
                    print(f"Error during profile removal: {remove_error}")
                    print(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to remove profile: {str(remove_error)}")
                    
        except Exception as e:
            print(f"Error removing profile {profile_name}: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to remove profile: {str(e)}")

    def init_menu(self):
        """init the file menu with a quit option"""
        try:
            menubar = self.menuBar()
            file_menu = menubar.addMenu('File')
            
            # core count override option
            core_count_action = QAction("Set Physical Core Count", self)
            core_count_action.triggered.connect(self.set_physical_core_count)
            file_menu.addAction(core_count_action)
            
            # separator
            file_menu.addSeparator()

            quit_action = QAction("Quit", self)
            quit_action.triggered.connect(self.confirm_quit)
            file_menu.addAction(quit_action)
        except Exception as e:
            print(f"Error initializing menu: {e}")
            print(traceback.format_exc())

    def set_physical_core_count(self):
        """allow manual override of the physical core count"""
        try:
            current_count = self.core_count
            detected_count = self.logical_core_count
            
            dialog = QDialog(self)
            dialog.setWindowTitle("Set Physical Core Count")
            
            layout = QVBoxLayout(dialog)
            
            # info label
            info_label = QLabel(f"Detected {detected_count} logical processors (with hyperthreading)", dialog)
            info_label.setStyleSheet("color: black;")
            layout.addWidget(info_label)
            
            # instruction
            instruction = QLabel("Enter the actual number of physical cores:", dialog)
            instruction.setStyleSheet("color: black;")
            layout.addWidget(instruction)
            
            # spinner for core count
            core_spinner = QSpinBox(dialog)
            core_spinner.setRange(1, detected_count)
            core_spinner.setValue(current_count)
            layout.addWidget(core_spinner)
            
            # note about physical vs logical cores
            note = QLabel("Note: Modern CPUs may show 2x logical processors with hyperthreading enabled", dialog)
            note.setStyleSheet("color: black; font-style: italic; font-size: 11px;")
            layout.addWidget(note)
            
            # buttons
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)
            
            # show dialog
            if dialog.exec_():
                new_count = core_spinner.value()
                if new_count != self.core_count:
                    self.core_count = new_count
                    print(f"Physical core count manually set to {new_count}")
                    
                    # update core cap if it exceeds the new count
                    if self.core_cap > new_count:
                        self.core_cap = new_count
                        self.manager.update_core_cap(new_count)
                    
                    # update max cores per profile if it exceeds the new count
                    if self.max_cores_per_profile > new_count:
                        self.max_cores_per_profile = new_count
                        self.manager.update_max_cores_per_profile(new_count)
                    
                    # update UI
                    self.update_core_cap_button()
                    
                    # show confirmation
                    QMessageBox.information(self, "Core Count Updated", 
                                        f"Physical core count set to {new_count}. Core limits updated accordingly.")
            
        except Exception as e:
            print(f"Error setting physical core count: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to set physical core count: {str(e)}")

    def confirm_quit(self):
        """show a confirmation dialog before quitting and stop all background processes if confirmed"""
        try:
            reply_box = QMessageBox(self)
            reply_box.setWindowTitle("Quit Confirmation")
            reply_box.setText("Are you sure you want to quit?")
            reply_box.setStyleSheet("QLabel { color: black; }") 
            reply_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.center_on_cursor(reply_box)
            reply = reply_box.exec_()

            if reply == QMessageBox.Yes:
                # stop all background processes before quitting
                self.manager.stop_all_profiles()
                self.queue_manager.stop()
                qApp.quit()
        except Exception as e:
            print(f"Error in confirm_quit: {e}")
            print(traceback.format_exc())
            # force quit if the normal path fails
            qApp.quit()

    def center_on_cursor(self, window=None):
        """center the given window or the main window on the monitor where the cursor is"""
        try:
            if window is None:
                window = self
            screen = QApplication.screenAt(QCursor.pos())
            if screen:  # check if screen is not None
                screen_geometry = screen.geometry()
                window.move(screen_geometry.center() - window.rect().center())
        except Exception as e:
            print(f"Error centering window: {e}")

    def closeEvent(self, event):
        """override close event to minimize to tray"""
        event.ignore()
        self.hide()  # explicitly hide the window
        try:
            self.tray_icon.showMessage(
                "File Processor App",
                "Application minimized to tray. Right-click the tray icon for options.",
                QSystemTrayIcon.Information,
                2000
            )
        except Exception as e:
            print(f"Error showing tray message: {e}")

    def set_network_folder(self):
        try:
            folder = QFileDialog.getExistingDirectory(self, "Select Network Folder")
            if folder:
                self.network_folder = folder
                self.folder_label.setText(f"Network Folder: {self.network_folder}")
                self.manager.update_network_folder(folder)
                self.load_profiles()
                self.update_profile_status_menu()
        except Exception as e:
            print(f"Error setting network folder: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to set network folder: {str(e)}")

    def load_profiles(self):
        """Load profiles with optimized change detection"""
        try:
            # Save current selection if any
            current_selection = None
            current_row = -1
            if self.job_list.selectedItems():
                current_item = self.job_list.currentItem()
                if current_item:
                    current_selection = current_item.text().split(" - ")[0]
                    current_row = self.job_list.row(current_item)
                        
            self.job_list.clear()
            
            # Get latest profile information
            profiles = self.manager.get_profiles_with_status()
            
            # Calculate total allocated cores for display
            total_allocated_cores = 0
            
            for profile, details in profiles.items():
                # Use display_status if available, otherwise fall back to status
                display_status = details.get('display_status', details['status'])
                cores_per_processor = details.get('cores_per_processor', 3)
                total_profile_cores = cores_per_processor * 2  # Total cores (JPEG + TIFF)
                
                # Update running total of allocated cores
                if display_status == 'Active' or display_status == 'Activating...':
                    total_allocated_cores += total_profile_cores
                
                # Display status and core allocation
                item_text = f"{profile} - {display_status} - {total_profile_cores} cores"
                
                self.job_list.addItem(item_text)
            
            # Sort items alphabetically
            self.job_list.sortItems()
            
            # Restore selection if possible
            if current_selection:
                for i in range(self.job_list.count()):
                    item = self.job_list.item(i)
                    if item and item.text().split(" - ")[0] == current_selection:
                        self.job_list.setCurrentRow(i)
                        break
                # If profile not found but there are items, try to select the same row or last item
                if not self.job_list.selectedItems() and self.job_list.count() > 0:
                    if current_row >= 0 and current_row < self.job_list.count():
                        self.job_list.setCurrentRow(current_row)
                    else:
                        self.job_list.setCurrentRow(self.job_list.count() - 1)
            
            # Update the status menu in the system tray
            self.update_profile_status_menu()
            
            # Update toggle button state
            self.update_toggle_button_state()
            
            # Update core cap display - show total allocated vs available
            self.update_core_cap_button(total_allocated_cores)
        except Exception as e:
            print(f"Error loading profiles: {e}")
            print(traceback.format_exc())

    def update_profile_list(self):
        """Optimized profile list update - only when actually needed"""
        try:
            # This will now only be called when something actually changes
            # (from user actions or background process state changes)
            
            current_selection = None
            current_row = -1
            if self.job_list.selectedItems():
                current_item = self.job_list.currentItem()
                if current_item:
                    current_selection = current_item.text().split(" - ")[0]
                    current_row = self.job_list.row(current_item)
            
            # Load profiles (this is the expensive operation)
            self.load_profiles()
            
            # Restore selection
            if current_selection:
                for i in range(self.job_list.count()):
                    item = self.job_list.item(i)
                    if item and item.text().split(" - ")[0] == current_selection:
                        self.job_list.setCurrentRow(i)
                        break
                if not self.job_list.selectedItems() and self.job_list.count() > 0:
                    if current_row >= 0 and current_row < self.job_list.count():
                        self.job_list.setCurrentRow(current_row)
                    else:
                        self.job_list.setCurrentRow(self.job_list.count() - 1)
        except Exception as e:
            print(f"Error updating profile list: {e}")

    def add_job(self):
        try:
            # get all current profiles to calculate available cores
            profiles = self.manager.get_profiles_with_status()
            
            # calculate total cores already allocated
            total_allocated_cores = 0
            for profile_name, details in profiles.items():
                cores_per_processor = details.get('cores_per_processor', 3)
                total_allocated_cores += cores_per_processor * 2  # total is JPEG + TIFF
            
            # calculate available cores (respect the core cap)
            available_cores = max(2, self.core_cap - total_allocated_cores)
            
            # calculate maximum cores allowed per profile based on constraints
            max_allowed_per_profile = min(
                self.max_cores_per_profile,  # user-defined max per profile
                self.core_count,             # physical constraint
                available_cores              # what's left from core cap
            )
            
            # default cores per processor (half of available, minimum 1)
            default_cores_per_processor = max(1, min(3, max_allowed_per_profile // 2))
            
            # pass both max_cores_per_profile and physical_cores
            dialog = ProfileNameDialog(max_allowed_per_profile, self.core_count, self)
            
            # set the default value in the spinner to a reasonable value
            dialog.cores_spin.setValue(default_cores_per_processor)
            
            if dialog.exec_():
                values = dialog.get_values()
                profile_name = values['profile_name']
                cores_per_processor = values['cores_per_processor']
                
                if profile_name:
                    print(f"Adding job profile: {profile_name} with {cores_per_processor} cores per processor")
                    # add the profile with specified cores per processor
                    new_profile = self.manager.add_profile(profile_name)
                    
                    # set the cores_per_processor attribute - in the profile settings
                    # this will be used to calculate the total cores for the profile (2 * cores_per_processor)
                    self.manager.update_cores_per_processor(profile_name, cores_per_processor)
                    
                    # after adding, explicitly call rebalance in the manager
                    self.manager.rebalance_cores()
                    
                    # update UI after adding profile
                    self.update_profile_list()
                    self.update_profile_status_menu()
                    print(f"Profile {profile_name} added successfully")
        except Exception as e:
            print(f"Error adding job: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to add job: {str(e)}")

    def remove_job(self):
        """remove the currently selected job profile"""
        try:
            # get the currently selected item
            selected_item = self.job_list.currentItem()
            if not selected_item:
                print("No profile selected for removal")
                return
                
            # extract profile name - handle the format "profile_name - status - cores cores"
            profile_text = selected_item.text()
            profile_name = profile_text.split(" - ")[0]
            
            print(f"Remove job button clicked for profile: {profile_name}")
                
            # show confirmation dialog
            reply_box = QMessageBox(self)
            reply_box.setWindowTitle("Confirm Removal")
            reply_box.setText(f"Are you sure you want to remove the profile '{profile_name}'?")
            reply_box.setStyleSheet("QLabel { color: black; }")
            reply_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.center_on_cursor(reply_box)
            reply = reply_box.exec_()
            
            if reply == QMessageBox.Yes:
                # make sure to stop the profile before removing it
                try:
                    if profile_name in self.manager.processes:
                        print(f"Stopping profile {profile_name} before removal")
                        self.manager.stop_processor(profile_name)
                except Exception as stop_error:
                    print(f"Error stopping profile before removal: {stop_error}")
                
                # now try to remove the profile
                try:
                    print(f"Removing profile {profile_name} directly from remove_job")
                    self.manager.remove_profile(profile_name)
                    print(f"Successfully removed profile {profile_name}")
                    
                    # update UI after removal
                    self.load_profiles()
                    self.update_profile_status_menu()
                    
                except Exception as remove_error:
                    print(f"Error removing profile: {remove_error}")
                    print(traceback.format_exc())
                    QMessageBox.critical(self, "Error", f"Failed to remove profile: {str(remove_error)}")
        except Exception as e:
            print(f"Error in remove_job: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to remove job: {str(e)}")

    def toggle_job_status(self):
        """toggle the status of the currently selected job profile"""
        try:
            selected_item = self.job_list.currentItem()
            if selected_item:
                profile_text = selected_item.text()
                profile_name = profile_text.split(" - ")[0]
                status = profile_text.split(" - ")[1].lower()
                
                print(f"Toggle button clicked for profile: {profile_name}, current status: {status}")
                
                # store selected row index
                current_row = self.job_list.row(selected_item)
                
                # directly perform the operation
                if status == "active":
                    print(f"Pausing profile {profile_name}")
                    self.manager.pause_profile(profile_name)
                else:
                    print(f"Starting profile {profile_name}")
                    self.manager.unpause_profile(profile_name)
                    
                # immediately update the UI after action
                self.update_profile_list()
                self.update_profile_status_menu()
                
                # try to restore selection
                if current_row < self.job_list.count():
                    self.job_list.setCurrentRow(current_row)
        except Exception as e:
            print(f"Error in toggle_job_status: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to toggle job status: {str(e)}")

    def set_core_cap(self):
        try:
            dialog = CoreCapDialog(self.core_cap, self.core_count, self.logical_core_count, self.max_cores_per_profile, self)
            if dialog.exec_():
                values = dialog.get_values()
                new_cap = values['core_cap']
                new_max_per_profile = values['max_per_profile']
                
                # ensure values don't exceed system constraints
                new_cap = min(new_cap, self.core_count)
                new_max_per_profile = min(new_max_per_profile, self.core_count)
                
                # check if the cap is being reduced
                cap_reduced = new_cap < self.core_cap
                
                # update settings
                self.core_cap = new_cap
                self.max_cores_per_profile = new_max_per_profile
                
                # update in manager
                self.manager.update_core_cap(new_cap)
                self.manager.update_max_cores_per_profile(new_max_per_profile)
                
                # update UI
                self.update_core_cap_button()
                
                # if reduced the cap, explicitly rebalance cores
                if cap_reduced:
                    self.manager.rebalance_cores()
                    self.update_profile_list()
                    self.update_profile_status_menu()
                    
                    # notify about the rebalancing
                    QMessageBox.information(self, "Core Settings Changed", 
                        "Core settings have been updated and profiles have been automatically rebalanced.")
                else:
                    # just inform about the change
                    QMessageBox.information(self, "Core Settings Changed", 
                        "Core settings have been updated. Changes will take effect when profiles are restarted.")
        except Exception as e:
            print(f"Error setting core cap: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to set core cap: {str(e)}")

    def update_core_cap_button(self, allocated_cores=None):
        try:
            if allocated_cores is None:
                # calculate total allocated cores
                profiles = self.manager.get_profiles_with_status()
                allocated_cores = 0
                for profile_name, details in profiles.items():
                    cores_per_processor = details.get('cores_per_processor', 3)
                    allocated_cores += cores_per_processor * 2
                    
            # show allocated vs available cores
            self.core_cap_btn.setText(
                f"Set Core Limits (Using: {allocated_cores}/{self.core_cap} cores, "
                f"Max/Profile: {self.max_cores_per_profile}, Physical: {self.core_count})"
            )
        except Exception as e:
            print(f"Error updating core cap button: {e}")
            # fallback to simpler display if error occurs
            self.core_cap_btn.setText(
                f"Set Core Limits (Total: {self.core_cap}, Max/Profile: {self.max_cores_per_profile})"
            )

    def start_all_profiles(self):
        """Force start all profiles regardless of their current status"""
        try:
            # show confirmation dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Start All Profiles")
            msg.setText("This will force-start ALL profiles regardless of their current status. Continue?")
            msg.setStyleSheet("QLabel { color: black; }")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.center_on_cursor(msg)
            reply = msg.exec_()
            
            if reply == QMessageBox.Yes:
                # Create progress dialog
                progress = QMessageBox(self)
                progress.setWindowTitle("Starting Profiles")
                progress.setText("Starting all profiles... This may take a moment.")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()  # Force UI update
                
                try:
                    # Start directly
                    print("Force-starting all profiles...")
                    profiles_count = len(self.manager.config.get('profiles', {}))
                    started_count = self.manager.start_all_profiles()
                    
                    # Update UI immediately after operation
                    self.update_profile_list()
                    self.update_profile_status_menu()
                    
                    # Show success message
                    progress.hide()
                    QMessageBox.information(self, "Profiles Started", 
                                        f"Successfully started {started_count} of {profiles_count} profiles.")
                    print(f"All profiles force-started: {started_count} of {profiles_count}")
                finally:
                    # Ensure progress dialog is closed
                    progress.hide()
                    
        except Exception as e:
            print(f"Error starting all profiles: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to start all profiles: {str(e)}")

    def stop_all_profiles(self):
        """stop all running profiles"""
        try:
            # show confirmation dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Stop All Profiles")
            msg.setText("Are you sure you want to stop all profiles?")
            msg.setStyleSheet("QLabel { color: black; }")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.center_on_cursor(msg)
            reply = msg.exec_()
            
            if reply == QMessageBox.Yes:
                # stop directly
                print("Stopping all profiles...")
                self.manager.stop_all_profiles()
                
                # update UI immediately after operation
                self.update_profile_list()
                self.update_profile_status_menu()
                print("All profiles stopped")
        except Exception as e:
            print(f"Error stopping all profiles: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to stop all profiles: {str(e)}")

    def toggle_job_status(self):
        """toggle the status of the currently selected job profile"""
        try:
            selected_item = self.job_list.currentItem()
            if selected_item:
                profile_text = selected_item.text()
                profile_name = profile_text.split(" - ")[0]
                status = profile_text.split(" - ")[1].lower()
                
                print(f"Toggle button clicked for profile: {profile_name}, current status: {status}")
                
                # store selected row index
                current_row = self.job_list.row(selected_item)
                
                # directly perform the operation without threading
                if status == "active":
                    # call the JobManager directly
                    print(f"Pausing profile {profile_name} directly from toggle_job_status")
                    self.manager.pause_profile(profile_name)
                else:
                    # call the JobManager directly
                    print(f"Starting profile {profile_name} directly from toggle_job_status")
                    self.manager.unpause_profile(profile_name)
                    
                # immediately update the UI
                self.update_profile_list()
                
                # try to restore selection
                if current_row < self.job_list.count():
                    self.job_list.setCurrentRow(current_row)
        except Exception as e:
            print(f"Error in toggle_job_status: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to toggle job status: {str(e)}")

    def set_core_cap(self):
        try:
            dialog = CoreCapDialog(self.core_cap, self.core_count, self.logical_core_count, self.max_cores_per_profile, self)
            if dialog.exec_():
                values = dialog.get_values()
                new_cap = values['core_cap']
                new_max_per_profile = values['max_per_profile']
                
                # ensure values don't exceed system constraints
                new_cap = min(new_cap, self.core_count)
                new_max_per_profile = min(new_max_per_profile, self.core_count)
                
                # check if the cap is being reduced
                cap_reduced = new_cap < self.core_cap
                
                # update settings
                self.core_cap = new_cap
                self.max_cores_per_profile = new_max_per_profile
                
                # update in manager
                self.manager.update_core_cap(new_cap)
                self.manager.update_max_cores_per_profile(new_max_per_profile)
                
                # update UI
                self.update_core_cap_button()
                
                # if reduced the cap, explicitly rebalance cores
                if cap_reduced:
                    self.manager.rebalance_cores()
                    self.update_profile_list()
                    
                    # notify about the rebalancing
                    QMessageBox.information(self, "Core Settings Changed", 
                        "Core settings have been updated and profiles have been automatically rebalanced.")
                else:
                    # just inform about the change
                    QMessageBox.information(self, "Core Settings Changed", 
                        "Core settings have been updated. Changes will take effect when profiles are restarted.")
        except Exception as e:
            print(f"Error setting core cap: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to set core cap: {str(e)}")
        
    def update_cores_per_processor(self, profile_name, cores_per_processor):
        """update the cores_per_processor setting for a specific profile"""
        try:
            if profile_name in self.config['profiles']:
                # ensure cores_per_processor is at least 1
                cores_per_processor = max(1, cores_per_processor)
                
                # update the cores_per_processor setting in the profile
                self.config['profiles'][profile_name]['cores_per_processor'] = cores_per_processor
                
                # save the configuration
                self.save_config()
                
                print(f"Updated {profile_name} to use {cores_per_processor} cores per processor")
                
                # restart the profile if it's active
                if self.config['profiles'][profile_name].get('status') == 'Active':
                    print(f"Restarting {profile_name} to apply core changes")
                    self.stop_processor(profile_name)
                    
                    # get paths from profile
                    jpeg_path = self.config['profiles'][profile_name].get('JPEG', '')
                    tiff_path = self.config['profiles'][profile_name].get('TIFF', '')
                    complete_path = self.config['profiles'][profile_name].get('COMPLETE', '')
                    
                    # restart with new core settings
                    self.start_processor(profile_name, "jpeg_processor.py", jpeg_path, complete_path, cores_per_processor)
                    self.start_processor(profile_name, "tiff_processor.py", tiff_path, complete_path, cores_per_processor)
                    
                return True
            return False
        except Exception as e:
            print(f"Error updating cores per processor: {e}")
            print(traceback.format_exc())
            return False

    def update_core_cap_button(self, allocated_cores=None):
        try:
            if allocated_cores is None:
                # calculate total allocated cores
                profiles = self.manager.get_profiles_with_status()
                allocated_cores = 0
                for profile_name, details in profiles.items():
                    cores_per_processor = details.get('cores_per_processor', 3)
                    allocated_cores += cores_per_processor * 2
                    
            # show allocated vs available cores
            self.core_cap_btn.setText(
                f"Set Core Limits (Using: {allocated_cores}/{self.core_cap} cores, "
                f"Max/Profile: {self.max_cores_per_profile}, Physical: {self.core_count})"
            )
        except Exception as e:
            print(f"Error updating core cap button: {e}")
            # fallback to simpler display if error occurs
            self.core_cap_btn.setText(
                f"Set Core Limits (Total: {self.core_cap}, Max/Profile: {self.max_cores_per_profile})"
            )

    def start_all_profiles(self):
        """Force start all profiles regardless of their current status"""
        try:
            # show confirmation dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Start All Profiles")
            msg.setText("This will force-start ALL profiles regardless of their current status. Continue?")
            msg.setStyleSheet("QLabel { color: black; }")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.center_on_cursor(msg)
            reply = msg.exec_()
            
            if reply == QMessageBox.Yes:
                # Create progress dialog
                progress = QMessageBox(self)
                progress.setWindowTitle("Starting Profiles")
                progress.setText("Starting all profiles... This may take a moment.")
                progress.setStandardButtons(QMessageBox.NoButton)
                progress.show()
                QApplication.processEvents()  # Force UI update
                
                try:
                    # Start directly without threading to ensure it works
                    print("Force-starting all profiles...")
                    profiles_count = len(self.manager.config.get('profiles', {}))
                    started_count = self.manager.start_all_profiles()
                    
                    # Update UI immediately
                    self.update_profile_list()
                    
                    # Show success message
                    progress.hide()
                    QMessageBox.information(self, "Profiles Started", 
                                        f"Successfully started {started_count} of {profiles_count} profiles.")
                    print(f"All profiles force-started: {started_count} of {profiles_count}")
                finally:
                    # Ensure progress dialog is closed
                    progress.hide()
                    
        except Exception as e:
            print(f"Error starting all profiles: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to start all profiles: {str(e)}")

    def stop_all_profiles(self):
        """stop all running profiles"""
        try:
            # show confirmation dialog
            msg = QMessageBox(self)
            msg.setWindowTitle("Stop All Profiles")
            msg.setText("Are you sure you want to stop all profiles?")
            msg.setStyleSheet("QLabel { color: black; }")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            self.center_on_cursor(msg)
            reply = msg.exec_()
            
            if reply == QMessageBox.Yes:
                # stop directly without threading to ensure it works
                print("Stopping all profiles...")
                self.manager.stop_all_profiles()
                # update UI immediately
                self.update_profile_list()
                print("All profiles stopped")
        except Exception as e:
            print(f"Error stopping all profiles: {e}")
            print(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to stop all profiles: {str(e)}")

if __name__ == "__main__":
    try:
        multiprocessing.freeze_support()  
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)  # prevent the app from quitting when the main window is closed
        
        # create and show the main window
        window = MainUI()
        window.show()
        
        # print confirmation that we've reached the event loop
        print("Entering Qt event loop")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical application error: {e}")
        print(traceback.format_exc())
        
        # create minimal error dialog if possible
        try:
            if 'app' in locals():
                QMessageBox.critical(None, "Fatal Error", 
                                 f"A fatal error occurred: {str(e)}")
        except:
            pass
        
        sys.exit(1)