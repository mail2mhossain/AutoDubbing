import os
import sys
import json
import shutil
from typing import List, Dict, Any, Optional
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QTabWidget, QProgressBar, 
    QLineEdit, QTableView, QHeaderView, QMessageBox, QTextEdit,
    QSplitter, QItemDelegate, QFrame, QSlider
)

# For VLC player integration
import vlc

# Import your existing workflow functions
from initialize import create_video_directory, create_temp_audio_folder
from audio_extractor import extract_audio 
from transcribe_audio import transcribe
from translate_transcription import load_translator, translate_srt
from translation_reviewer import review_translation, regenerate_translated_srt
from audio_generator import text_to_speech
from vocal_separator import separate_vocals_with_demucs
from dubbed_audio_generator import generate_dubbed_audio
from dubbing_n_embedding import create_dubbed_video

from contextlib import contextmanager
from accelerate.utils import release_memory
import torch, gc
from decouple import config


OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)


class TranslationItemDelegate(QItemDelegate):
    """Custom delegate for the translation editor."""
    def createEditor(self, parent, option, index):
        if index.column() == 4:  # translated_text column
            editor = QTextEdit(parent)
            return editor
        return super().createEditor(parent, option, index)
    
    def setEditorData(self, editor, index):
        if index.column() == 4:  # translated_text column
            text = index.model().data(index, Qt.DisplayRole)
            editor.setText(text)
        else:
            super().setEditorData(editor, index)
    
    def setModelData(self, editor, model, index):
        if index.column() == 4:  # translated_text column
            model.setData(index, editor.toPlainText(), Qt.EditRole)
        else:
            super().setModelData(editor, model, index)


class TranslationReviewWindow(QMainWindow):
    """Window for reviewing translations."""
    translationAccepted = Signal(str)  # Signal to emit when translation is accepted
    
    def __init__(self, diarization_file: str, parent=None):
        super().__init__(parent)
        self.diarization_file = diarization_file
        self.diarization_data = []
        
        self.setWindowTitle("AutoDubbing: Translation Review")
        self.setMinimumSize(1000, 600)
        
        self.initUI()
        self.loadDiarizationData()
    
    def initUI(self):
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        # Create splitter for two-panel view
        splitter = QSplitter(Qt.Horizontal)
        
        # Table view for translation data
        self.tableView = QTableView()
        self.tableView.setSelectionBehavior(QTableView.SelectRows)
        self.tableView.setAlternatingRowColors(True)
        self.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableView.verticalHeader().setVisible(False)
        
        # Set up delegate for editing
        self.delegate = TranslationItemDelegate()
        self.tableView.setItemDelegate(self.delegate)
        
        # Create model for table
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Index", "Start", "End", "Original Text", "Translated Text"])
        self.tableView.setModel(self.model)
        
        # Detail view showing original and translated text for the selected row
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        
        self.originalTextLabel = QLabel("Original Text:")
        self.originalTextEdit = QTextEdit()
        self.originalTextEdit.setReadOnly(True)
        
        self.translatedTextLabel = QLabel("Translated Text:")
        self.translatedTextEdit = QTextEdit()
        
        detail_layout.addWidget(self.originalTextLabel)
        detail_layout.addWidget(self.originalTextEdit)
        detail_layout.addWidget(self.translatedTextLabel)
        detail_layout.addWidget(self.translatedTextEdit)
        
        # Connect selection change to update detail view
        self.tableView.selectionModel().selectionChanged.connect(self.updateDetailView)
        
        # When edited in detail view, update model
        self.translatedTextEdit.textChanged.connect(self.updateModelFromDetailView)
        
        # Add views to splitter
        splitter.addWidget(self.tableView)
        splitter.addWidget(detail_widget)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.continueButton = QPushButton("Continue")
        self.continueButton.clicked.connect(self.saveDiarizationData)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.close)
        
        button_layout.addWidget(self.cancelButton)
        button_layout.addWidget(self.continueButton)
        
        layout.addLayout(button_layout)
        
        self.setCentralWidget(central_widget)
    
    
    def enableVideoPreview(self):
        """Enable video preview with VLC players"""
        # Clean up existing tabs
        while self.videoTabs.count() > 0:
            self.videoTabs.removeTab(0)
        
        # Create original video tab
        original_video_container = QWidget()
        original_video_layout = QVBoxLayout(original_video_container)
        
        # Create frame for VLC player
        self.originalVideoFrame = QFrame()
        self.originalVideoFrame.setMinimumHeight(450)
        self.originalVideoFrame.setStyleSheet("background-color: black;")
        
        # Create VLC instance and player
        self.vlc_instance = vlc.Instance()
        self.originalPlayer = self.vlc_instance.media_player_new()
        
        # Set the window ID where to render VLC's video output
        if sys.platform.startswith('linux'):  # for Linux
            self.originalPlayer.set_xwindow(int(self.originalVideoFrame.winId()))
        elif sys.platform == "win32":  # for Windows
            self.originalPlayer.set_hwnd(int(self.originalVideoFrame.winId()))
        elif sys.platform == "darwin":  # for MacOS
            self.originalPlayer.set_nsobject(int(self.originalVideoFrame.winId()))
        
        original_video_layout.addWidget(self.originalVideoFrame, 1)
        
        # Original video controls container
        self.originalControlsContainer = QWidget()
        original_controls_container_layout = QVBoxLayout(self.originalControlsContainer)
        original_controls_container_layout.setContentsMargins(0, 0, 0, 0)
        original_controls_container_layout.setSpacing(5)
        
        # Original video controls
        original_controls = QHBoxLayout()
        original_controls.setSpacing(5)
        
        self.originalPlayButton = QPushButton("Play")
        self.originalPlayButton.clicked.connect(self.playOriginalVideo)
        self.originalPlayButton.setMaximumWidth(60)
        
        self.originalPauseButton = QPushButton("Pause")
        self.originalPauseButton.clicked.connect(self.pauseOriginalVideo)
        self.originalPauseButton.setMaximumWidth(60)
        
        self.originalStopButton = QPushButton("Stop")
        self.originalStopButton.clicked.connect(self.stopOriginalVideo)
        self.originalStopButton.setMaximumWidth(60)
        
        original_controls.addWidget(self.originalPlayButton)
        original_controls.addWidget(self.originalPauseButton)
        original_controls.addWidget(self.originalStopButton)
        original_controls.addStretch()
        
        # Combined controls layout for position and volume
        combined_controls = QHBoxLayout()
        combined_controls.setSpacing(10)
        
        # Position control
        position_layout = QHBoxLayout()
        position_layout.setSpacing(5)
        self.originalPositionLabel = QLabel("0:00 / 0:00")
        self.originalPositionSlider = QSlider(Qt.Horizontal)
        self.originalPositionSlider.setRange(0, 1000)
        self.originalPositionSlider.sliderMoved.connect(self.setOriginalPosition)
        
        position_layout.addWidget(QLabel("Pos:"))
        position_layout.addWidget(self.originalPositionSlider, 1)
        position_layout.addWidget(self.originalPositionLabel)
        
        # Volume control
        volume_layout = QHBoxLayout()
        volume_layout.setSpacing(5)
        volume_layout.setContentsMargins(5, 5, 5, 5) 
        self.originalVolumeLabel = QLabel("Vol:  ")

        self.originalVolumeSlider = QSlider(Qt.Horizontal)
        self.originalVolumeSlider.setRange(0, 100)
        self.originalVolumeSlider.setValue(80)
        self.originalVolumeSlider.valueChanged.connect(self.setOriginalVolume)
        
        volume_layout.addWidget(self.originalVolumeLabel)
        volume_layout.addWidget(self.originalVolumeSlider, 1)
        
        # Add position and volume to the combined layout
        combined_controls.addLayout(position_layout, 3)
        combined_controls.addLayout(volume_layout, 1)
        
        # Add layouts to original controls container
        original_controls_container_layout.addLayout(original_controls)
        original_controls_container_layout.addLayout(combined_controls)
        
        # Add controls container to main layout
        original_video_layout.addWidget(self.originalControlsContainer, 0)
        
        # Create dubbed video tab
        dubbed_video_container = QWidget()
        dubbed_video_layout = QVBoxLayout(dubbed_video_container)
        
        # Create frame for VLC player
        self.dubbedVideoFrame = QFrame()
        self.dubbedVideoFrame.setMinimumHeight(450)
        self.dubbedVideoFrame.setStyleSheet("background-color: black;")
        
        # Create VLC player for dubbed video
        self.dubbedPlayer = self.vlc_instance.media_player_new()
        
        # Set the window ID where to render VLC's video output
        if sys.platform.startswith('linux'):  # for Linux
            self.dubbedPlayer.set_xwindow(int(self.dubbedVideoFrame.winId()))
        elif sys.platform == "win32":  # for Windows
            self.dubbedPlayer.set_hwnd(int(self.dubbedVideoFrame.winId()))
        elif sys.platform == "darwin":  # for MacOS
            self.dubbedPlayer.set_nsobject(int(self.dubbedVideoFrame.winId()))
        
        dubbed_video_layout.addWidget(self.dubbedVideoFrame, 1)
        
        # Dubbed video controls container
        self.dubbedControlsContainer = QWidget()
        dubbed_controls_container_layout = QVBoxLayout(self.dubbedControlsContainer)
        dubbed_controls_container_layout.setContentsMargins(0, 0, 0, 0)
        dubbed_controls_container_layout.setSpacing(5)
        
        # Dubbed video controls
        dubbed_controls = QHBoxLayout()
        dubbed_controls.setSpacing(5)
        
        self.dubbedPlayButton = QPushButton("Play")
        self.dubbedPlayButton.clicked.connect(self.playDubbedVideo)
        self.dubbedPlayButton.setMaximumWidth(60)
        
        self.dubbedPauseButton = QPushButton("Pause")
        self.dubbedPauseButton.clicked.connect(self.pauseDubbedVideo)
        self.dubbedPauseButton.setMaximumWidth(60)
        
        self.dubbedStopButton = QPushButton("Stop")
        self.dubbedStopButton.clicked.connect(self.stopDubbedVideo)
        self.dubbedStopButton.setMaximumWidth(60)
        
        dubbed_controls.addWidget(self.dubbedPlayButton)
        dubbed_controls.addWidget(self.dubbedPauseButton)
        dubbed_controls.addWidget(self.dubbedStopButton)
        dubbed_controls.addStretch()
        
        # Combined controls layout for position and volume
        dubbed_combined_controls = QHBoxLayout()
        dubbed_combined_controls.setSpacing(10)
        
        # Position control
        dubbed_position_layout = QHBoxLayout()
        dubbed_position_layout.setSpacing(5)
        self.dubbedPositionLabel = QLabel("0:00 / 0:00")
        self.dubbedPositionSlider = QSlider(Qt.Horizontal)
        self.dubbedPositionSlider.setRange(0, 1000)
        self.dubbedPositionSlider.sliderMoved.connect(self.setDubbedPosition)
        
        dubbed_position_layout.addWidget(QLabel("Pos:"))
        dubbed_position_layout.addWidget(self.dubbedPositionSlider, 1)
        dubbed_position_layout.addWidget(self.dubbedPositionLabel)
        
        # Volume control
        dubbed_volume_layout = QHBoxLayout()
        dubbed_volume_layout.setSpacing(5)
        dubbed_volume_layout.setContentsMargins(5, 5, 5, 5) 
        self.dubbedVolumeLabel = QLabel("Vol:")
        self.dubbedVolumeLabel.setFixedWidth(35)  # Set fixed width for the label
        self.dubbedVolumeLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # Right-align the text
        self.dubbedVolumeSlider = QSlider(Qt.Horizontal)
        self.dubbedVolumeSlider.setRange(0, 100)
        self.dubbedVolumeSlider.setValue(80)
        self.dubbedVolumeSlider.valueChanged.connect(self.setDubbedVolume)
        
        dubbed_volume_layout.addWidget(self.dubbedVolumeLabel)
        dubbed_volume_layout.addWidget(self.dubbedVolumeSlider, 1)
        
        # Add position and volume to the combined layout
        dubbed_combined_controls.addLayout(dubbed_position_layout, 3)
        dubbed_combined_controls.addLayout(dubbed_volume_layout, 1)
        
        # Add layouts to dubbed controls container
        dubbed_controls_container_layout.addLayout(dubbed_controls)
        dubbed_controls_container_layout.addLayout(dubbed_combined_controls)
        
        # Add controls container to main layout
        dubbed_video_layout.addWidget(self.dubbedControlsContainer, 0)
        
        # Add the tabs
        self.videoTabs.addTab(original_video_container, "Original Video")
        self.videoTabs.addTab(dubbed_video_container, "Dubbed Video")
        
        # Load videos if paths are available
        if hasattr(self, 'video_file') and self.video_file:
            media = self.vlc_instance.media_new(self.video_file)
            self.originalPlayer.set_media(media)
            self.enableOriginalVideoControls(True)
        
        if hasattr(self, 'dubbed_video_path') and self.dubbed_video_path:
            media = self.vlc_instance.media_new(self.dubbed_video_path)
            self.dubbedPlayer.set_media(media)
            self.enableDubbedVideoControls(True)
            
        self.videoControlsEnabled = True

    def playOriginalVideo(self):
        """Play original video"""
        if hasattr(self, 'originalPlayer'):
            self.originalPlayer.play()
    
    def pauseOriginalVideo(self):
        """Pause original video"""
        if hasattr(self, 'originalPlayer'):
            self.originalPlayer.pause()
    
    def stopOriginalVideo(self):
        """Stop original video"""
        if hasattr(self, 'originalPlayer'):
            self.originalPlayer.stop()
    
    def playDubbedVideo(self):
        """Play dubbed video"""
        if hasattr(self, 'dubbedPlayer'):
            self.dubbedPlayer.play()
    
    def pauseDubbedVideo(self):
        """Pause dubbed video"""
        if hasattr(self, 'dubbedPlayer'):
            self.dubbedPlayer.pause()
    
    def stopDubbedVideo(self):
        """Stop dubbed video"""
        if hasattr(self, 'dubbedPlayer'):
            self.dubbedPlayer.stop()
    
    def setOriginalVolume(self, value):
        """Set original video volume"""
        if hasattr(self, 'originalPlayer'):
            self.originalPlayer.audio_set_volume(value)
    
    def setDubbedVolume(self, value):
        """Set dubbed video volume"""
        if hasattr(self, 'dubbedPlayer'):
            self.dubbedPlayer.audio_set_volume(value)


    def playVideo(self):
        """Play video in VLC player"""
        try:
            # Create VLC instance
            self.instance = vlc.Instance()
            self.player = self.instance.media_player_new()
            
            # Load media
            media = self.instance.media_new(self.video_file)
            self.player.set_media(media)
            
            # Set video output to the QVideoWidget
            self.player.set_xwindow(self.videoWidget.winId())
            
            # Play the video
            self.player.play()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to play video: {str(e)}")
    
    def loadDiarizationData(self):
        """Load diarization data from file"""
        try:
            with open(self.diarization_file, "r", encoding="utf-8") as f:
                self.diarization_data = json.load(f)
            
            self.populateTable()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load diarization data: {str(e)}")
    
    def populateTable(self):
        """Populate table with diarization data"""
        self.model.removeRows(0, self.model.rowCount())
        
        for item in self.diarization_data:
            index_item = QStandardItem(str(item['index']))
            start_item = QStandardItem(str(item['start']))
            end_item = QStandardItem(str(item['end']))
            text_item = QStandardItem(item['text'])
            translated_item = QStandardItem(item['translated_text'])
            
            # Make only translated_text editable
            index_item.setEditable(False)
            start_item.setEditable(False)
            end_item.setEditable(False)
            text_item.setEditable(False)
            
            self.model.appendRow([index_item, start_item, end_item, text_item, translated_item])
        
        # Auto-select first row
        if self.model.rowCount() > 0:
            self.tableView.selectRow(0)
    
    def updateDetailView(self):
        """Update detail view based on selected row"""
        indexes = self.tableView.selectionModel().selectedRows()
        if indexes:
            # Disconnect the textChanged signal temporarily to prevent triggering updateModelFromDetailView
            self.translatedTextEdit.textChanged.disconnect(self.updateModelFromDetailView)
            
            row = indexes[0].row()
            original_text = self.model.item(row, 3).text()
            translated_text = self.model.item(row, 4).text()
            
            self.originalTextEdit.setText(original_text)
            self.translatedTextEdit.setText(translated_text)
            
            # Store current row for updating
            self.current_row = row
            
            # Reconnect the textChanged signal
            self.translatedTextEdit.textChanged.connect(self.updateModelFromDetailView)
            
    def updateModelFromDetailView(self):
        """Update model data when detail view is edited"""
        if hasattr(self, 'current_row'):
            translated_text = self.translatedTextEdit.toPlainText()
            
            # Update the model item (which updates the table view)
            self.model.item(self.current_row, 4).setText(translated_text)
            
            # Also update the diarization_data directly
            index = int(self.model.item(self.current_row, 0).text())
            for item in self.diarization_data:
                if item['index'] == index:
                    item['translated_text'] = translated_text
                    break
    
    def saveDiarizationData(self):
        """Save updated diarization data back to file"""
        try:
            # Update diarization data from model
            for row in range(self.model.rowCount()):
                index = int(self.model.item(row, 0).text())
                # Find matching item in diarization_data
                for item in self.diarization_data:
                    if item['index'] == index:
                        item['translated_text'] = self.model.item(row, 4).text()
            
            # Save updated data
            with open(self.diarization_file, "w", encoding="utf-8") as f:
                json.dump(self.diarization_data, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "Success", "Translation data saved successfully!")
            
            # Emit signal that translation was accepted
            self.translationAccepted.emit(self.diarization_file)
            self.close()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save diarization data: {str(e)}")


class WorkerThread(QThread):
    """Worker thread to run processing steps"""
    progress = Signal(int, str)
    finished = Signal(str, str)  # result, error message
    
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None

        # Check if progress_callback should be added
        if func == create_dubbed_video and 'progress_callback' not in kwargs:
            self.kwargs['progress_callback'] = self.report_progress
    
    def report_progress(self, percent, message):
        """Report progress back to the main thread"""
        self.progress.emit(percent, message)

    def run(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
            self.finished.emit(str(self.result), "")
        except Exception as e:
            self.finished.emit("", str(e))



class DubbingApp(QMainWindow):
    """Main application window"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDubbing")
        self.setMinimumSize(1200, 800)
        
        self.video_file = None
        self.video_dir = None
        self.temp_audio_dir = None
        self.audio_path = None
        self.diarization_file = None
        self.dubbed_video_path = None
        self.current_step = 0
        self.total_steps = 10
        
        self.initUI()
    
    def initUI(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # File selection area
        file_layout = QHBoxLayout()
        self.filePathEdit = QLineEdit()
        self.filePathEdit.setReadOnly(True)
        self.filePathEdit.setPlaceholderText("Select a video file...")
        
        self.browseButton = QPushButton("Browse")
        self.browseButton.clicked.connect(self.browseFile)
        
        file_layout.addWidget(QLabel("Video File:"))
        file_layout.addWidget(self.filePathEdit)
        file_layout.addWidget(self.browseButton)
        
        main_layout.addLayout(file_layout)
        
        # Progress area
        progress_layout = QHBoxLayout()
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        
        self.statusLabel = QLabel("Ready")
        self.statusLabel.setStyleSheet("color: blue;")
        
        progress_layout.addWidget(QLabel("Progress:"))
        progress_layout.addWidget(self.progressBar)
        progress_layout.addWidget(self.statusLabel)
        
        main_layout.addLayout(progress_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()  # Add stretch before buttons to push them to the right
        self.startButton = QPushButton("Start Dubbing")
        self.startButton.clicked.connect(self.startDubbingProcess)
        self.startButton.setEnabled(False)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.cancelProcess)
        self.cancelButton.setEnabled(False)
        
        action_layout.addStretch()  # Add stretch before buttons to push them to the right
        action_layout.addWidget(self.startButton)
        action_layout.addWidget(self.cancelButton)
        
        main_layout.addLayout(action_layout)
        
        # Video tabs
        self.videoTabs = QTabWidget()
        self.videoTabs.currentChanged.connect(self.onTabChanged)
        
        # Create original video tab
        original_video_container = QWidget()
        original_video_layout = QVBoxLayout(original_video_container)
        
        # Create frame for VLC player
        self.originalVideoFrame = QFrame()
        self.originalVideoFrame.setMinimumHeight(450)
        self.originalVideoFrame.setStyleSheet("background-color: black;")
        
        # Create VLC instance and player
        self.vlc_instance = vlc.Instance()
        self.originalPlayer = self.vlc_instance.media_player_new()
        
        # Set the window ID where to render VLC's video output
        if sys.platform.startswith('linux'):  # for Linux
            self.originalPlayer.set_xwindow(int(self.originalVideoFrame.winId()))
        elif sys.platform == "win32":  # for Windows
            self.originalPlayer.set_hwnd(int(self.originalVideoFrame.winId()))
        elif sys.platform == "darwin":  # for MacOS
            self.originalPlayer.set_nsobject(int(self.originalVideoFrame.winId()))
        
        original_video_layout.addWidget(self.originalVideoFrame, 1)
        
        # Original video controls container
        self.originalControlsContainer = QWidget()
        original_controls_container_layout = QVBoxLayout(self.originalControlsContainer)
        original_controls_container_layout.setContentsMargins(0, 0, 0, 0)
        original_controls_container_layout.setSpacing(5)
        
        # Original video controls
        original_controls = QHBoxLayout()
        original_controls.setSpacing(5)
        
        self.originalPlayButton = QPushButton("Play")
        self.originalPlayButton.clicked.connect(self.playOriginalVideo)
        self.originalPlayButton.setMaximumWidth(60)
        
        self.originalPauseButton = QPushButton("Pause")
        self.originalPauseButton.clicked.connect(self.pauseOriginalVideo)
        self.originalPauseButton.setMaximumWidth(60)
        
        self.originalStopButton = QPushButton("Stop")
        self.originalStopButton.clicked.connect(self.stopOriginalVideo)
        self.originalStopButton.setMaximumWidth(60)
        
        original_controls.addWidget(self.originalPlayButton)
        original_controls.addWidget(self.originalPauseButton)
        original_controls.addWidget(self.originalStopButton)
        original_controls.addStretch()
        
        # Combined controls layout for position and volume
        combined_controls = QHBoxLayout()
        combined_controls.setSpacing(10)
        
        # Position control
        position_layout = QHBoxLayout()
        position_layout.setSpacing(5)
        self.originalPositionLabel = QLabel("0:00 / 0:00")
        self.originalPositionSlider = QSlider(Qt.Horizontal)
        self.originalPositionSlider.setRange(0, 1000)
        self.originalPositionSlider.sliderMoved.connect(self.setOriginalPosition)
        
        position_layout.addWidget(QLabel("Pos:"))
        position_layout.addWidget(self.originalPositionSlider, 1)
        position_layout.addWidget(self.originalPositionLabel)
        
        # Volume control
        volume_layout = QHBoxLayout()
        volume_layout.setSpacing(0)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        self.originalVolumeLabel = QLabel("Vol:")
        self.originalVolumeLabel.setFixedWidth(30)  # Set fixed width for the label
        self.originalVolumeLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # Right-align the text
        self.originalVolumeSlider = QSlider(Qt.Horizontal)
        self.originalVolumeSlider.setRange(0, 100)
        self.originalVolumeSlider.setValue(80)
        self.originalVolumeSlider.valueChanged.connect(self.setOriginalVolume)
        
        volume_layout.addWidget(self.originalVolumeLabel)
        volume_layout.addWidget(self.originalVolumeSlider, 1)
        
        # Add position and volume to the combined layout
        combined_controls.addLayout(position_layout, 3)
        combined_controls.addLayout(volume_layout, 1)
        
        # Add layouts to original controls container
        original_controls_container_layout.addLayout(original_controls)
        original_controls_container_layout.addLayout(combined_controls)
        
        # Add controls container to main layout
        original_video_layout.addWidget(self.originalControlsContainer, 0)
        
        # Create dubbed video tab
        dubbed_video_container = QWidget()
        dubbed_video_layout = QVBoxLayout(dubbed_video_container)
        
        # Create frame for VLC player
        self.dubbedVideoFrame = QFrame()
        self.dubbedVideoFrame.setMinimumHeight(450)
        self.dubbedVideoFrame.setStyleSheet("background-color: black;")
        
        # Create VLC player for dubbed video
        self.dubbedPlayer = self.vlc_instance.media_player_new()
        
        # Set the window ID where to render VLC's video output
        if sys.platform.startswith('linux'):  # for Linux
            self.dubbedPlayer.set_xwindow(int(self.dubbedVideoFrame.winId()))
        elif sys.platform == "win32":  # for Windows
            self.dubbedPlayer.set_hwnd(int(self.dubbedVideoFrame.winId()))
        elif sys.platform == "darwin":  # for MacOS
            self.dubbedPlayer.set_nsobject(int(self.dubbedVideoFrame.winId()))
        
        dubbed_video_layout.addWidget(self.dubbedVideoFrame, 1)
        
        # Dubbed video controls container
        self.dubbedControlsContainer = QWidget()
        dubbed_controls_container_layout = QVBoxLayout(self.dubbedControlsContainer)
        dubbed_controls_container_layout.setContentsMargins(0, 0, 0, 0)
        dubbed_controls_container_layout.setSpacing(5)
        
        # Dubbed video controls
        dubbed_controls = QHBoxLayout()
        dubbed_controls.setSpacing(5)
        
        self.dubbedPlayButton = QPushButton("Play")
        self.dubbedPlayButton.clicked.connect(self.playDubbedVideo)
        self.dubbedPlayButton.setMaximumWidth(60)
        
        self.dubbedPauseButton = QPushButton("Pause")
        self.dubbedPauseButton.clicked.connect(self.pauseDubbedVideo)
        self.dubbedPauseButton.setMaximumWidth(60)
        
        self.dubbedStopButton = QPushButton("Stop")
        self.dubbedStopButton.clicked.connect(self.stopDubbedVideo)
        self.dubbedStopButton.setMaximumWidth(60)
        
        dubbed_controls.addWidget(self.dubbedPlayButton)
        dubbed_controls.addWidget(self.dubbedPauseButton)
        dubbed_controls.addWidget(self.dubbedStopButton)
        dubbed_controls.addStretch()
        
        # Combined controls layout for position and volume
        dubbed_combined_controls = QHBoxLayout()
        dubbed_combined_controls.setSpacing(10)
        
        # Position control
        dubbed_position_layout = QHBoxLayout()
        dubbed_position_layout.setSpacing(5)
        self.dubbedPositionLabel = QLabel("0:00 / 0:00")
        self.dubbedPositionSlider = QSlider(Qt.Horizontal)
        self.dubbedPositionSlider.setRange(0, 1000)
        self.dubbedPositionSlider.sliderMoved.connect(self.setDubbedPosition)
        
        dubbed_position_layout.addWidget(QLabel("Pos:"))
        dubbed_position_layout.addWidget(self.dubbedPositionSlider, 1)
        dubbed_position_layout.addWidget(self.dubbedPositionLabel)
        
        # Volume control
        dubbed_volume_layout = QHBoxLayout()
        dubbed_volume_layout.setSpacing(0)
        dubbed_volume_layout.setContentsMargins(0, 0, 0, 0)
        self.dubbedVolumeLabel = QLabel("Vol:")
        self.dubbedVolumeLabel.setFixedWidth(30)  # Set fixed width for the label
        self.dubbedVolumeLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # Right-align the text
        self.dubbedVolumeSlider = QSlider(Qt.Horizontal)
        self.dubbedVolumeSlider.setRange(0, 100)
        self.dubbedVolumeSlider.setValue(80)
        self.dubbedVolumeSlider.valueChanged.connect(self.setDubbedVolume)
        
        dubbed_volume_layout.addWidget(self.dubbedVolumeLabel)
        dubbed_volume_layout.addWidget(self.dubbedVolumeSlider, 1)
        
        # Add position and volume to the combined layout
        dubbed_combined_controls.addLayout(dubbed_position_layout, 3)
        dubbed_combined_controls.addLayout(dubbed_volume_layout, 1)
        
        # Add layouts to dubbed controls container
        dubbed_controls_container_layout.addLayout(dubbed_controls)
        dubbed_controls_container_layout.addLayout(dubbed_combined_controls)
        
        # Add controls container to main layout
        dubbed_video_layout.addWidget(self.dubbedControlsContainer, 0)
        
        # Add the tabs
        self.videoTabs.addTab(original_video_container, "Original Video")
        self.videoTabs.addTab(dubbed_video_container, "Dubbed Video")
        
        main_layout.addWidget(self.videoTabs)
        
        self.setCentralWidget(central_widget)
        
        # Initially disable video controls until a video is loaded
        self.enableVideoControls(False)
        
        # Initialize the visibility of controls based on the current tab
        # Set initial visibility - important!
        if self.videoTabs.currentIndex() == 0:
            self.originalControlsContainer.setVisible(True)
            self.dubbedControlsContainer.setVisible(False)
        else:
            self.originalControlsContainer.setVisible(False)
            self.dubbedControlsContainer.setVisible(True)
    
    def browseFile(self):
        """Open file dialog to select video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)"
        )
        
        if file_path:
            self.video_file = file_path
            self.filePathEdit.setText(file_path)
            self.startButton.setEnabled(True)
            
            # Set up original video
            media = self.vlc_instance.media_new(self.video_file)
            self.originalPlayer.set_media(media)
            self.enableOriginalVideoControls(True)
    
    def enableVideoControls(self, enable=True):
        """Enable or disable video controls"""
        # Enable/disable the controls based on parameter
        self.enableOriginalVideoControls(enable)
        self.enableDubbedVideoControls(enable)
        
        # Make sure the visibility still respects the current tab
        current_tab = self.videoTabs.currentIndex()
        self.onTabChanged(current_tab)
    
    def enableOriginalVideoControls(self, enable=True):
        """Enable or disable original video controls"""
        self.originalPlayButton.setEnabled(enable)
        self.originalPauseButton.setEnabled(enable)
        self.originalStopButton.setEnabled(enable)
        self.originalPositionSlider.setEnabled(enable)
        self.originalVolumeSlider.setEnabled(enable)
    
    def enableDubbedVideoControls(self, enable=True):
        """Enable or disable dubbed video controls"""
        self.dubbedPlayButton.setEnabled(enable)
        self.dubbedPauseButton.setEnabled(enable)
        self.dubbedStopButton.setEnabled(enable)
        self.dubbedPositionSlider.setEnabled(enable)
        self.dubbedVolumeSlider.setEnabled(enable)
    
    def playOriginalVideo(self):
        """Play original video"""
        self.originalPlayer.play()
    
    def pauseOriginalVideo(self):
        """Pause original video"""
        self.originalPlayer.pause()
    
    def stopOriginalVideo(self):
        """Stop original video"""
        self.originalPlayer.stop()
    
    def playDubbedVideo(self):
        """Play dubbed video"""
        self.dubbedPlayer.play()
    
    def pauseDubbedVideo(self):
        """Pause dubbed video"""
        self.dubbedPlayer.pause()
    
    def stopDubbedVideo(self):
        """Stop dubbed video"""
        self.dubbedPlayer.stop()
    
    def setOriginalVolume(self, value):
        """Set original video volume"""
        self.originalPlayer.audio_set_volume(value)
    
    def setDubbedVolume(self, value):
        """Set dubbed video volume"""
        self.dubbedPlayer.audio_set_volume(value)
    
    def setOriginalPosition(self, position):
        """Set position in original video based on slider value"""
        if self.originalPlayer.is_playing() or self.originalPlayer.get_length() > 0:
            # Convert position from slider range (0-1000) to VLC position (0.0-1.0)
            vlc_position = position / 1000.0
            self.originalPlayer.set_position(vlc_position)
    
    def setDubbedPosition(self, position):
        """Set position in dubbed video based on slider value"""
        if self.dubbedPlayer.is_playing() or self.dubbedPlayer.get_length() > 0:
            # Convert position from slider range (0-1000) to VLC position (0.0-1.0)
            vlc_position = position / 1000.0
            self.dubbedPlayer.set_position(vlc_position)
    
    def updateProgress(self, value, message):
        """Update progress bar and status message"""
        self.progressBar.setValue(value)
        self.statusLabel.setText(message)
    
    def startDubbingProcess(self):
        """Start the dubbing process"""
        self.current_step = 0
        self.startButton.setEnabled(False)
        self.cancelButton.setEnabled(True)
        
        # Initialize progress
        self.updateProgress(0, "Starting...")
        
        # Create directories (Step 1)
        self.initializeDirectories()
    
    def initializeDirectories(self):
        """Initialize directories for processing (Step 1)"""
        try:
            # Setup directories
            print("Creating directories...")
            self.updateProgress(10, "Creating directories...")
            
            self.video_dir = create_video_directory(self.video_file)
            self.temp_audio_dir = create_temp_audio_folder(self.video_dir)
            
            # Continue to next step - Extract audio
            self.extractAudio()
        except Exception as e:
            self.handleError(f"Failed to initialize directories: {str(e)}")
    
    def extractAudio(self):
        """Extract audio from video (Step 2)"""
        print("Extracting audio...")
        self.updateProgress(20, "Extracting audio...")
        
        # Set up paths
        video_file_name = os.path.splitext(os.path.basename(self.video_file))[0]
        self.audio_path = os.path.join(self.temp_audio_dir, f"{video_file_name}.wav")
        
        if os.path.exists(self.audio_path):
            self.transcribeAudio()
            return
        # Run in worker thread
        self.worker = WorkerThread(extract_audio, self.video_file, self.audio_path)
        self.worker.finished.connect(lambda result, error: self.onExtractAudioFinished(result, error))
        self.worker.start()
    
    def onExtractAudioFinished(self, result, error):
        """Handle audio extraction completion"""
        if error:
            self.handleError(f"Failed to extract audio: {error}")
            return
        
        # Continue to transcription
        self.transcribeAudio()
    
    def transcribeAudio(self):
        """Transcribe audio (Step 3)"""
        print("Transcribing audio...")
        self.updateProgress(30, "Transcribing audio...")
        
        # Set up paths
        video_file_name = os.path.splitext(os.path.basename(self.video_file))[0]
        srt_filename = os.path.join(self.video_dir, f"{video_file_name}_en.srt")
        model_size = "large-v3"  # Use your preferred model size
        
        if os.path.exists(srt_filename):
            json_filename = os.path.splitext(srt_filename)[0] + "_diarization.json"
            self.diarization_file = json_filename
            self.translateText()
            return
        # Run in worker thread
        self.worker = WorkerThread(transcribe, self.audio_path, srt_filename, model_size)
        self.worker.finished.connect(lambda result, error: self.onTranscribeFinished(result, error))
        self.worker.start()
    
    def onTranscribeFinished(self, result, error):
        """Handle transcription completion"""
        if error:
            self.handleError(f"Failed to transcribe audio: {error}")
            return
        
        self.diarization_file = result
        
        # Continue to translation
        self.translateText()
    
    def translateText(self):
        """Translate text (Step 4)"""
        print("Translating text...")
        self.updateProgress(40, "Translating text...")
        
        # Set up paths
        video_file_name = os.path.splitext(os.path.basename(self.video_file))[0]
        translated_srt_filename = os.path.join(self.video_dir, f"{video_file_name}_bn.srt")
        
        if os.path.exists(translated_srt_filename):
            self.reviewTranslation()
            return
        # We need to use a context manager for the translator
        @contextmanager
        def translator_ctx():
            tr = load_translator()
            try:
                yield tr
            finally:
                release_memory(tr.model)  
                del tr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect() 
                
                if os.path.exists("offload"):
                    shutil.rmtree("offload")
        
        def translate_with_context():
            with translator_ctx() as translator:
                return translate_srt(self.diarization_file, translator)
        
        # Run in worker thread
        self.worker = WorkerThread(translate_with_context)
        self.worker.finished.connect(lambda result, error: self.onTranslateFinished(result, error))
        self.worker.start()
    
    def onTranslateFinished(self, result, error):
        """Handle translation completion"""
        if error:
            self.handleError(f"Failed to translate text: {error}")
            return
        
        self.translated_srt_filename = result
        # Continue to Manual translation review
        self.reviewTranslation()
    
    def reviewTranslation(self):
        """Review translation (Step 5)"""
        print("ChatGPT Reviewing translation...")
        self.updateProgress(45, "ChatGPT Reviewing translation...")
        
        # ChatGPT Translation Review
        if OPENAI_API_KEY is not None:
            review_translation(self.diarization_file)
        else:
            print("OPENAI_API_KEY is not set. Skipping translation review.")
        
        print("Opening translation review...")
        self.updateProgress(50, "Opening translation review...")
        # Open translation review window
        self.review_window = TranslationReviewWindow(self.diarization_file, self)
        self.review_window.translationAccepted.connect(self.onTranslationReviewed)
        self.review_window.showMaximized()
    
    @Slot(str)
    def onTranslationReviewed(self, diarization_file):
        """Handle translation review completion"""
        # Regenerate translated SRT file
        video_file_name = os.path.splitext(os.path.basename(self.video_file))[0]
        translated_srt_filename = os.path.join(self.video_dir, f"{video_file_name}_bn.srt")
        regenerate_translated_srt(diarization_file, translated_srt_filename)
        
        # Continue to text-to-speech
        self.textToSpeech()
    
    def textToSpeech(self):
        """Generate speech from text (Step 6)"""
        print("Generating speech from text...")
        self.updateProgress(60, "Generating speech from text...")
        
        # Run in worker thread
        self.worker = WorkerThread(text_to_speech, self.video_dir, self.diarization_file)
        self.worker.finished.connect(lambda result, error: self.onTextToSpeechFinished(result, error))
        self.worker.start()
    
    def onTextToSpeechFinished(self, result, error):
        """Handle text-to-speech completion"""
        if error:
            self.handleError(f"Failed to generate speech: {error}")
            return
        
        # Continue to vocal separation
        self.separateVocals()
    
    def separateVocals(self):
        """Separate vocals from audio (Step 7)"""
        print("Separating vocals from audio...")
        self.updateProgress(70, "Separating vocals from audio...")
        
        demucs_output_dir = os.path.join(self.temp_audio_dir, "demucs_output")
        
        # Run in worker thread
        self.worker = WorkerThread(separate_vocals_with_demucs, self.audio_path, demucs_output_dir)
        self.worker.finished.connect(lambda result, error: self.onSeparateVocalsFinished(result, error))
        self.worker.start()
    
    def onSeparateVocalsFinished(self, result, error):
        """Handle vocal separation completion"""
        if error:
            self.handleError(f"Failed to separate vocals: {error}")
            return
        
        # Parse the result which should be a tuple (vocals_path, no_vocals_path)
        result_tuple = eval(result)  # Convert string to tuple
        vocals_path, no_vocals_path = result_tuple
        
        # Store no_vocals_path for the next step
        self.no_vocals_path = no_vocals_path
        
        # Continue to dubbed audio generation
        self.generateDubbedAudio()
    
    def generateDubbedAudio(self):
        """Generate dubbed audio (Step 8)"""
        print("Generating dubbed audio...")
        self.updateProgress(80, "Generating dubbed audio...")
        
        # if os.path.exists(self.dubbed_vocals_audio_file):
        #     self.createDubbedVideo()
        #     return
        # Run in worker thread
        self.worker = WorkerThread(generate_dubbed_audio, self.diarization_file, self.no_vocals_path)
        self.worker.finished.connect(lambda result, error: self.onGenerateDubbedAudioFinished(result, error))
        self.worker.start()
    
    def onGenerateDubbedAudioFinished(self, result, error):
        """Handle dubbed audio generation completion"""
        if error:
            self.handleError(f"Failed to generate dubbed audio: {error}")
            return
        
        # Store dubbed audio path
        self.dubbed_vocals_audio_file = result
        
        # Continue to dubbed video creation
        self.createDubbedVideo()
    
    def createDubbedVideo(self):
        """Create dubbed video (Step 9)"""
        print("Creating dubbed video...")
        self.updateProgress(90, "Creating dubbed video...")
        
        # Setup paths
        video_file_name = os.path.splitext(os.path.basename(self.video_file))[0]
        srt_filename = os.path.join(self.video_dir, f"{video_file_name}_en.srt")
        translated_srt_filename = os.path.join(self.video_dir, f"{video_file_name}_bn.srt")
        self.dubbed_video_path = os.path.join(self.video_dir, f"{video_file_name}_dubbed.mp4")
        
        # Run in worker thread
        self.worker = WorkerThread(
            create_dubbed_video,
            self.video_file,
            self.dubbed_vocals_audio_file,
            srt_filename,
            translated_srt_filename,
            self.dubbed_video_path
        )
        # Connect progress signal to update the UI
        self.worker.progress.connect(self.updateFFmpegProgress)
        self.worker.finished.connect(lambda result, error: self.onCreateDubbedVideoFinished(result, error))
        self.worker.start()

    def updateFFmpegProgress(self, percent, message):
        """Update progress specifically for FFmpeg operations"""
        # Calculate overall progress (90% start + up to 10% for FFmpeg)
        # This ensures we keep the 90% base progress and add the FFmpeg portion
        overall_percent = 90 + int(percent / 10)
        self.updateProgress(overall_percent, message)

    
    def onCreateDubbedVideoFinished(self, result, error):
        """Handle dubbed video creation completion"""
        if error:
            self.handleError(f"Failed to create dubbed video: {error}")
            return
        
        # Process complete
        self.updateProgress(100, "Dubbing process complete!")
        self.finishProcess()
    
    def finishProcess(self):
        """Finish the dubbing process"""
        print("Dubbing process complete!")
        self.startButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        
        # Set up dubbed video player
        if hasattr(self, 'dubbedPlayer') and self.dubbed_video_path:
            media = self.vlc_instance.media_new(self.dubbed_video_path)
            self.dubbedPlayer.set_media(media)
            self.enableDubbedVideoControls(True)
            
            # Switch to dubbed video tab
            self.videoTabs.setCurrentIndex(1)
        
        QMessageBox.information(self, "Success", f"Video dubbing process completed successfully!\nDubbed video saved to:\n{self.dubbed_video_path}")
    
    def cancelProcess(self):
        """Cancel the ongoing process"""
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.terminate()
        
        self.updateProgress(0, "Process canceled.")
        self.startButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
    
    def handleError(self, message):
        """Handle error during processing"""
        self.updateProgress(0, f"Error: {message}")
        self.startButton.setEnabled(True)
        self.cancelButton.setEnabled(False)
        QMessageBox.critical(self, "Error", message)
    
    def onTabChanged(self, index):
        """Handle tab change events to show/hide appropriate controls"""
        if index == 0:  # Original Video tab
            self.originalControlsContainer.setVisible(True)
            self.dubbedControlsContainer.setVisible(False)
        else:  # Dubbed Video tab
            self.originalControlsContainer.setVisible(False)
            self.dubbedControlsContainer.setVisible(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DubbingApp()
    window.showMaximized()
    sys.exit(app.exec())