document.addEventListener('DOMContentLoaded', function() {
    const chatForm = document.getElementById('chatForm');
    const chatboxMessages = document.getElementById('chatbox-messages');
    const chatboxContainer = document.getElementById('chatbox-container');
    const chatboxHeader = document.getElementById('chatbox-header');
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
        if (message.toLowerCase() === 'clear') {
            clearChatbox();
            return;
        }
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
    function clearChatbox() {
        chatboxMessages.innerHTML = '';
    }
    // Make the chatbox draggable
    chatboxHeader.addEventListener('mousedown', function(e) {
        let offsetX = e.clientX - chatboxContainer.getBoundingClientRect().left;
        let offsetY = e.clientY - chatboxContainer.getBoundingClientRect().top;

        function onMouseMove(e) {
            let newLeft = e.clientX - offsetX;
            let newTop = e.clientY - offsetY;

            // Ensure the chatbox stays within the viewport boundaries
            if (newLeft < 0) newLeft = 0;
            if (newTop < 0) newTop = 0;
            if (newLeft + chatboxContainer.offsetWidth > window.innerWidth) {
                newLeft = window.innerWidth - chatboxContainer.offsetWidth;
            }
            if (newTop + chatboxContainer.offsetHeight > window.innerHeight) {
                newTop = window.innerHeight - chatboxContainer.offsetHeight;
            }

            chatboxContainer.style.left = `${newLeft}px`;
            chatboxContainer.style.top = `${newTop}px`;
        }

        function onMouseUp() {
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
        }

        document.addEventListener('mousemove', onMouseMove);
        document.addEventListener('mouseup', onMouseUp);
    });
    // chatboxContainer.addEventListener('resize', function() {
    //     const rect = chatboxContainer.getBoundingClientRect();
    //     if (rect.right > window.innerWidth) {
    //         chatboxContainer.style.width = `${window.innerWidth - rect.left}px`;
    //     }
    //     if (rect.bottom > window.innerHeight) {
    //         chatboxContainer.style.height = `${window.innerHeight - rect.top}px`;
    //     }
    // });
});