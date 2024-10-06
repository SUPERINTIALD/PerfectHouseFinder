document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const chatboxMessages = document.getElementById('chatbox-messages');
    const chatboxContainer = document.getElementById('chatbox-container');
    const openChatboxButton = document.getElementById('open-chatbox');
    const closeChatboxButton = document.getElementById('close-chatbox');

    openChatboxButton.addEventListener('click', function() {
        chatboxContainer.classList.toggle('open');
    });

    closeChatboxButton.addEventListener('click', function() {
        chatboxContainer.classList.remove('open');
    });

    chatForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const messageInput = document.getElementById('message');
        const message = messageInput.value;
        
        // Display the user's message in the chatbox
        addMessageToChat('User', message);
        
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
            // Display the chatbot's response in the chatbox
            addMessageToChat('Assistant', data.results[0]);
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });

    function addMessageToChat(sender, message) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.innerHTML = `<strong>${sender}:</strong> ${message}`;
        chatboxMessages.appendChild(messageElement);
        chatboxMessages.scrollTop = chatboxMessages.scrollHeight;
    }
});