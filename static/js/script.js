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

    output.style.width = '100%';
    output.style.height = '400px';
    output.style.overflowY = 'auto';

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    let mediaRecorder = null;
    let audioChunks = [];

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
                        const data = await response.json();
                        console.log('Transcription response:', data);

                        if (data.transcription) {
                            input.value = data.transcription;
                            console.log('Textarea updated with:', data.transcription);
                            recordingStatus.style.display = 'none';
                        } else {
                            recordingStatus.textContent = data.error || 'Transcription failed';
                            console.error('Transcription error:', data.error);
                            setTimeout(() => { recordingStatus.style.display = 'none'; }, 3000);
                        }
                    } catch (error) {
                        console.error('Fetch error:', error);
                        recordingStatus.textContent = 'Error transcribing audio';
                        setTimeout(() => { recordingStatus.style.display = 'none'; }, 3000);
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
                console.error('Microphone error:', error);
                recordingStatus.textContent = 'Microphone access denied';
                recordingStatus.style.display = 'block';
                setTimeout(() => { recordingStatus.style.display = 'none'; }, 3000);
            }
        } else {
            console.log('Stopping recording');
            mediaRecorder.stop();
            recordBtn.textContent = 'Record';
            recordBtn.classList.remove('btn-danger');
            recordBtn.classList.add('btn-outline-primary');
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
        console.error('Qwen error:', error);
        const errorDiv = document.createElement('div');
        errorDiv.className = 'chat-message bot-message';
        errorDiv.textContent = 'Error communicating with server';
        output.appendChild(errorDiv);
    }

    output.scrollTop = output.scrollHeight;
}