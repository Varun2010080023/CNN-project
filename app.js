const fileInput = document.querySelector('#photo-input');
const dropZone = document.querySelector('#drop-zone');
const identifyButton = document.querySelector('#identify-button');
const buttonLabel = document.querySelector('#button-label');
const errorMessage = document.querySelector('#error-message');
const preview = document.querySelector('#preview');
const previewImage = document.querySelector('#preview-image');
const fileName = document.querySelector('#file-name');
const emptyResult = document.querySelector('#empty-result');
const resultContent = document.querySelector('#result-content');

let selectedFile = null;
let previewUrl = null;
let facts = [];
let factIndex = 0;

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = !message;
}

function setFile(file) {
  showError('');
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
    selectedFile = null;
    preview.hidden = true;
    identifyButton.disabled = true;
    showError('Please choose a JPG, PNG, or WebP photo.');
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    selectedFile = null;
    preview.hidden = true;
    identifyButton.disabled = true;
    showError('Please choose an image smaller than 10 MB.');
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  selectedFile = file;
  previewImage.src = previewUrl;
  fileName.textContent = file.name;
  preview.hidden = false;
  identifyButton.disabled = false;
  emptyResult.hidden = false;
  resultContent.hidden = true;
}

fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

for (const eventName of ['dragenter', 'dragover']) {
  dropZone.addEventListener(eventName, event => {
    event.preventDefault();
    dropZone.classList.add('dragging');
  });
}
for (const eventName of ['dragleave', 'dragend']) {
  dropZone.addEventListener(eventName, () => dropZone.classList.remove('dragging'));
}
dropZone.addEventListener('drop', event => {
  event.preventDefault();
  dropZone.classList.remove('dragging');
  setFile(event.dataTransfer.files[0]);
});

function displayFact() {
  document.querySelector('#fact-text').textContent = facts.length
    ? facts[factIndex]
    : 'Fun facts for this fruit have not been added yet.';
  document.querySelector('#next-fact').hidden = facts.length < 2;
}

document.querySelector('#next-fact').addEventListener('click', () => {
  const next = Math.floor(Math.random() * (facts.length - 1));
  factIndex = next >= factIndex ? next + 1 : next;
  displayFact();
});

function showResult(data) {
  const fruit = data.fruit;
  document.querySelector('#result-emoji').textContent = fruit === 'apple' ? '🍎' : '🍌';
  document.querySelector('#result-title').textContent = fruit[0].toUpperCase() + fruit.slice(1);

  facts = Array.isArray(data.info.facts)
    ? data.info.facts.filter(fact => typeof fact === 'string' && fact.trim())
    : [];
  factIndex = facts.length ? Math.floor(Math.random() * facts.length) : 0;
  displayFact();

  const joke = typeof data.info.joke === 'string' ? data.info.joke.trim() : '';
  document.querySelector('#joke-text').textContent = joke;
  document.querySelector('#joke-section').hidden = !joke;

  const nutrition = data.info.nutrition || {};
  const nutrients = nutrition.nutrients || {};
  const entries = Object.entries(nutrients);
  document.querySelector('#serving').textContent = entries.length
    ? `Reference values for ${nutrition.serving}. These are not measured from your photo.`
    : 'Nutrition information for this fruit has not been added yet.';
  const list = document.querySelector('#nutrition-list');
  list.replaceChildren();
  for (const [name, amount] of entries) {
    const term = document.createElement('dt');
    term.textContent = name;
    const definition = document.createElement('dd');
    definition.textContent = String(amount);
    list.append(term, definition);
  }
  document.querySelector('#nutrition-source').textContent = nutrition.source
    ? `Source: ${nutrition.source}` : '';
  emptyResult.hidden = true;
  resultContent.hidden = false;
}

identifyButton.addEventListener('click', async () => {
  if (!selectedFile) return;
  const submittedFile = selectedFile;
  showError('');
  identifyButton.disabled = true;
  buttonLabel.textContent = 'Taking a closer look…';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': submittedFile.type},
      body: submittedFile
    });
    const data = await response.json();
    if (selectedFile !== submittedFile) return;
    if (!response.ok) throw new Error(data.error || 'The photo could not be identified.');
    showResult(data);
  } catch (error) {
    if (selectedFile === submittedFile) showError(error.message || 'Something went wrong. Please try again.');
  } finally {
    buttonLabel.textContent = 'Identify my fruit';
    identifyButton.disabled = !selectedFile;
  }
});

window.addEventListener('beforeunload', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
});
