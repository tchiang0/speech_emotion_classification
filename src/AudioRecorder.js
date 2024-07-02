import React, { useEffect, useState } from 'react';
import { useVoiceVisualizer, VoiceVisualizer } from 'react-voice-visualizer';
import './AudioRecorder.css';

const AudioRecorder = () => {
  const recorderControls = useVoiceVisualizer();
  const {
    recordedBlob,
    error,
    clearCanvas
  } = recorderControls;
  const [uploadStatus, setUploadStatus] = useState('');

  useEffect(() => {
    if (!recordedBlob) return;
    console.log(recordedBlob);
  }, [recordedBlob, error]);

  useEffect(() => {
    if (!error) return;
    console.error(error);
  }, [error]);

  const saveRecording = async () => {
    if (!recordedBlob) return;

    const formData = new FormData();
    formData.append('audio', recordedBlob, 'recording.webm');

    try {
      const response = await fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData,
      });
      if (response.ok){
        const result = await response.json();
        console.log(result);
        setUploadStatus('File uploaded successfully: ' + result.filename);
      }
      else {
      setUploadStatus('File upload failed');
      }
    } catch (err) {
      setUploadStatus('File upload failed');
      console.error('Error uploading file:', err);
    }
  };

  const clearRecording = () => {
    clearCanvas();
    setUploadStatus('');
  };

  return (
    <div className="audio-recorder">
      <VoiceVisualizer controls={recorderControls} height={200} width="100%" backgroundColor="#000000" />
      <button onClick={saveRecording} disabled={!recordedBlob}>
          Save Recording
      </button>
      <button onClick={clearRecording} disabled={!recordedBlob}>
          Clear Recording
      </button>
      {uploadStatus && <div className="upload-status">{uploadStatus}</div>}
    </div>
  );
};

export default AudioRecorder;
