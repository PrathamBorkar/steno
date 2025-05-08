// DOM Elements
const mainContent = document.getElementById('main-content');
const encodeForm = document.getElementById('encode-form');
const decodeForm = document.getElementById('decode-form');
const encryptTab = document.getElementById('encrypt-tab');
const decryptTab = document.getElementById('decrypt-tab');
const encryptSection = document.getElementById('encrypt-section');
const decryptSection = document.getElementById('decrypt-section');
const resultDisplay = document.getElementById('result-display');
const imageComparison = document.getElementById('image-comparison');
const originalPreview = document.getElementById('original-preview');
const encodedPreview = document.getElementById('encoded-preview');
const downloadLink = document.getElementById('download-link');
const messageTextarea = document.getElementById('encode-message');
const charCountSpan = document.getElementById('char-count');
const patchworkDecodeExtra = document.getElementById('patchwork-decode-extra');

// Tab switching handlers
encryptTab.addEventListener('click', () => {
    encryptTab.classList.add("bg-lime-600", "text-white");
    encryptTab.classList.remove("bg-gray-700", "text-gray-300");
    decryptTab.classList.remove("bg-lime-600", "text-white");
    decryptTab.classList.add("bg-gray-700", "text-gray-300");
    encryptSection.classList.remove("hidden");
    decryptSection.classList.add("hidden");
    resultDisplay.textContent = "";
});

decryptTab.addEventListener('click', () => {
    decryptTab.classList.add("bg-lime-600", "text-white");
    decryptTab.classList.remove("bg-gray-700", "text-gray-300");
    encryptTab.classList.remove("bg-lime-600", "text-white");
    encryptTab.classList.add("bg-gray-700", "text-gray-300");
    decryptSection.classList.remove("hidden");
    encryptSection.classList.add("hidden");
    resultDisplay.textContent = "";
});

// Character count for message textarea
messageTextarea.addEventListener('input', () => {
    charCountSpan.textContent = `${messageTextarea.value.length}/500 CHARACTERS`;
});

// Remove the old radio button handler and replace with simplified version
const methodRadios = decodeForm.querySelectorAll('input[name="method"]');
methodRadios.forEach((radio) => {
    radio.addEventListener("change", () => {
        if (radio.value === "patchwork" && radio.checked) {
            patchworkDecodeExtra.classList.remove("hidden");
        } else {
            patchworkDecodeExtra.classList.add("hidden");
        }
    });
});

// Handle encode form submission
encodeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultDisplay.textContent = "Encoding...";
    imageComparison.classList.add("hidden");

    // Display original image immediately after selection
    const selectedFile = encodeForm.image.files[0];
    if (selectedFile) {
        const objectUrl = URL.createObjectURL(selectedFile);
        originalPreview.src = objectUrl;
        originalPreview.onload = () => {
            URL.revokeObjectURL(objectUrl); // Clean up the URL after image loads
        };
    }

    const formData = new FormData(encodeForm);

    try {
        const response = await fetch("/encode", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        // In the encode form submission handler, after successful encoding
        if (data.success) {
            resultDisplay.textContent = `Encoding successful!`;
        
            // Display encoded image
            encodedPreview.src = `/get_image/${data.output_filename}`;
            downloadLink.href = `/get_image/${data.output_filename}`;
            downloadLink.download = data.output_filename;
            imageComparison.classList.remove("hidden");
        
            if (formData.get("method") === "patchwork") {
                resultDisplay.textContent += `\nPairs: ${data.pairs}\nAlpha: ${data.alpha}`;
            }
            
            // Reset the form
            encodeForm.reset();
            charCountSpan.textContent = "0/500 CHARACTERS";
        } else {
            resultDisplay.textContent = `Encoding failed: ${data.error}`;
        }
    } catch (error) {
        resultDisplay.textContent = `Error: ${error.message}`;
    }
});

// Add decode form submission handler
decodeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    resultDisplay.textContent = "Decoding...";

    const formData = new FormData(decodeForm);

    try {
        const response = await fetch("/decode", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        // In the decode form submission handler, after successful decoding
        if (data.success) {
            resultDisplay.textContent = `Decoded Message:\n${data.message}`;
            
            // Reset the form
            decodeForm.reset();
        } else {
            resultDisplay.textContent = `Decoding failed: ${data.error}`;
        }
    } catch (error) {
        resultDisplay.textContent = `Error: ${error.message}`;
    }
});
