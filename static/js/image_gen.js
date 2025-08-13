document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');
    const textarea = document.querySelector('textarea[name="symptoms"]');
    const submitButton = document.querySelector('form button');
    const errorContainer = document.querySelector('.error') || createErrorContainer();
    let spinner = null;

    function createErrorContainer() {
        const container = document.createElement('p');
        container.className = 'error';
        container.style.display = 'none';
        form.insertAdjacentElement('afterend', container);
        return container;
    }

    function createSpinner() {
        const spinner = document.createElement('span');
        spinner.className = 'spinner';
        spinner.style.marginLeft = '1rem';
        return spinner;
    }

    function validateInput() {
        const symptoms = textarea.value.trim();
        if (!symptoms) {
            errorContainer.textContent = 'Please enter symptoms.';
            errorContainer.style.display = 'block';
            return false;
        }
        if (symptoms.length < 10) {
            errorContainer.textContent = 'Symptoms must be at least 10 characters long.';
            errorContainer.style.display = 'block';
            return false;
        }
        errorContainer.style.display = 'none';
        return true;
    }

    function showLoading() {
        submitButton.disabled = true;
        submitButton.textContent = 'Generating...';
        submitButton.style.backgroundColor = '#007A8A';
        submitButton.style.cursor = 'not-allowed';
        if (!spinner) {
            spinner = createSpinner();
            submitButton.insertAdjacentElement('afterend', spinner);
        }
    }

    function resetLoading() {
        submitButton.disabled = false;
        submitButton.textContent = 'Generate Image';
        submitButton.style.backgroundColor = '#00A3B3';
        submitButton.style.cursor = 'pointer';
        if (spinner) {
            spinner.remove();
            spinner = null;
        }
    }

    form.addEventListener('submit', function (event) {
        if (!validateInput()) {
            event.preventDefault();
            return;
        }
        showLoading();
        setTimeout(() => {
            const serverError = document.querySelector('.error');
            if (serverError && serverError.style.display !== 'none') {
                resetLoading();
                const retryButton = document.createElement('button');
                retryButton.textContent = 'Retry';
                retryButton.className = 'retry-button';
                retryButton.addEventListener('click', () => form.submit());
                serverError.insertAdjacentElement('afterend', retryButton);
            } else {
                resetLoading(); // Reset even if no error to ensure UI updates
            }
        }, 2000);
    });

    if (errorContainer.textContent) {
        resetLoading();
    }
});