document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('segmentation-form');
    const fileInput = document.getElementById('image');

    form.addEventListener('submit', (e) => {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert('Please select an image');
            return;
        }
        // Provide feedback during processing
        const button = form.querySelector('button');
        button.textContent = 'Processing...';
        button.disabled = true;
    });

    // Reset button text on page load if form was submitted
    const button = form.querySelector('button');
    button.textContent = 'Segment Image';
    button.disabled = false;
});