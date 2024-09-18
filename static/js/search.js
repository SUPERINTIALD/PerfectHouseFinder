function performSearch() {
    const query = document.getElementById('search-bar').value;
    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        const resultsDiv = document.getElementById('search-results');
        resultsDiv.innerHTML = '';
        data.results.forEach(result => {
            const resultElement = document.createElement('div');
            resultElement.classList.add('result-item');
            resultElement.innerText = result;
            resultsDiv.appendChild(resultElement);
        });
    })
    .catch(error => console.error('Error:', error));
}

document.getElementById('search-button').addEventListener('click', performSearch);

document.getElementById('search-bar').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        performSearch();
    }
});