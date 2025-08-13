document.addEventListener('DOMContentLoaded', () => {
    // Add smooth scroll to card buttons
    document.querySelectorAll('.btn-primary').forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const href = button.getAttribute('href');
            window.location.href = href;
        });
    });
});