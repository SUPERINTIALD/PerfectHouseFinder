document.getElementById('chatForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const messageInput = document.getElementById('message');
    const message = messageInput.value;
    
    // Display the message in the chatbox
    const chatbox = document.getElementById('chatbox');
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.textContent = message;
    chatbox.appendChild(userMessage);
    
    // Clear the input field
    messageInput.value = '';
    
    // Send the message to the server and get the NLP response
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query: message })
    })
    .then(response => response.json())
    .then(data => {
        // Display the NLP response in the chatbox
        const nlpResponse = document.createElement('div');
        nlpResponse.className = 'nlp-response';
        nlpResponse.textContent = data.results[0];
        chatbox.appendChild(nlpResponse);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});