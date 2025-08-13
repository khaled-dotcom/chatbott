document.addEventListener('DOMContentLoaded', () => {
    console.log('Chatbot page loaded');
    const input = document.getElementById('user-input');
    const output = document.getElementById('chat-output');
    const recordBtn = document.getElementById('record-btn');
    const recordingStatus = document.getElementById('recording-status');

    if (!input || !output || !recordBtn) {
        console.error('Missing DOM elements:', { input, output, recordBtn });
        return;
    }

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            stopRecording();
            sendMessage();
        }
    });

    let mediaRecorder = null;
    let audioChunks = [];

    function stopRecording() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            console.log('Stopping recording');
            mediaRecorder.stop();
            recordBtn.textContent = 'Record';
            recordBtn.classList.remove('btn-danger');
            recordBtn.classList.add('btn-outline-primary');
        }
    }

    recordBtn.addEventListener('click', async () => {
        console.log('Record button clicked, state:', mediaRecorder ? mediaRecorder.state : 'none');
        if (!mediaRecorder || mediaRecorder.state === 'inactive') {
            try {
                console.log('Requesting microphone access');
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                console.log('Microphone access granted');
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = (e) => {
                    audioChunks.push(e.data);
                    console.log('Audio chunk received');
                };

                mediaRecorder.onstop = async () => {
                    console.log('Recording stopped, sending audio');
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'recording.wav');

                    recordingStatus.textContent = 'Transcribing...';
                    recordingStatus.style.display = 'block';

                    try {
                        const response = await fetch('/chatbot/transcribe', {
                            method: 'POST',
                            body: formData
                        });
                        console.log('Transcription response status:', response.status, response.statusText);
                        const contentType = response.headers.get('content-type');
                        console.log('Content-Type:', contentType);

                        if (!contentType || !contentType.includes('application/json')) {
                            const text = await response.text();
                            console.error('Non-JSON response:', text.slice(0, 200));
                            recordingStatus.textContent = `Error: Server returned non-JSON response (${response.status})`;
                            setTimeout(() => { recordingStatus.style.display = 'none'; }, 5000);
                            return;
                        }

                        const data = await response.json();
                        console.log('Transcription response data:', data);

                        if (data.transcription) {
                            input.value = data.transcription;
                            console.log('Textarea updated with:', data.transcription);
                            recordingStatus.style.display = 'none';
                        } else {
                            recordingStatus.textContent = data.error || `Transcription failed (Status: ${response.status})`;
                            console.error('Transcription error:', data.error);
                            setTimeout(() => { recordingStatus.style.display = 'none'; }, 5000);
                        }
                    } catch (error) {
                        console.error('Transcription fetch error:', error.message);
                        recordingStatus.textContent = `Error transcribing audio: ${error.message}`;
                        setTimeout(() => { recordingStatus.style.display = 'none'; }, 5000);
                    }

                    stream.getTracks().forEach(track => track.stop());
                };

                mediaRecorder.start();
                recordBtn.textContent = 'Stop';
                recordBtn.classList.add('btn-danger');
                recordBtn.classList.remove('btn-outline-primary');
                recordingStatus.textContent = 'Recording...';
                recordingStatus.style.display = 'block';
            } catch (error) {
                console.error('Microphone error:', error.message);
                recordingStatus.textContent = 'Microphone access denied';
                recordingStatus.style.display = 'block';
                setTimeout(() => { recordingStatus.style.display = 'none'; }, 5000);
            }
        } else {
            stopRecording();
        }
    });
});

async function sendMessage() {
    console.log('Send button clicked');
    const input = document.getElementById('user-input');
    const output = document.getElementById('chat-output');
    const message = input.value.trim();

    if (!message) {
        console.log('Empty message, ignoring');
        return;
    }

    const userDiv = document.createElement('div');
    userDiv.className = 'chat-message user-message';
    userDiv.textContent = message;
    output.appendChild(userDiv);

    input.value = '';

    try {
        console.log('Sending message:', message);
        const response = await fetch('/chatbot/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ instruction: message })
        });
        const data = await response.json();
        console.log('Qwen response:', data);

        const botDiv = document.createElement('div');
        botDiv.className = 'chat-message bot-message';
        botDiv.innerHTML = marked.parse(data.response || data.error);
        output.appendChild(botDiv);
    } catch (error) {
        console.error('Qwen error:', error.message);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chat-message bot-message';
        errorDiv.textContent = 'Error communicating with server';
        output.appendChild(errorDiv);
    }

    output.scrollTop = output.scrollHeight;
}