// document.getElementById('chatForm').addEventListener('submit', function(event) {
//     event.preventDefault();
    
//     const messageInput = document.getElementById('message');
//     const message = messageInput.value;
    
//     // Display the message in the chatbox
//     const chatbox = document.getElementById('chatbox');
//     const userMessage = document.createElement('div');
//     userMessage.className = 'user-message';
//     userMessage.textContent = message;
//     chatbox.appendChild(userMessage);
    
//     // Clear the input field
//     messageInput.value = '';
    
//     // Send the message to the server and get the NLP response
//     fetch('/chat', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({ query: message })
//     })
//     .then(response => response.json())
//     .then(data => {
//         // Display the NLP response in the chatbox
//         const nlpResponse = document.createElement('div');
//         nlpResponse.className = 'nlp-response';
//         nlpResponse.textContent = data.results[0];
//         chatbox.appendChild(nlpResponse);
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// });

// document.getElementById('chatForm').addEventListener('submit', function(event) {
//     event.preventDefault();
    
//     const messageInput = document.getElementById('message');
//     const message = messageInput.value;
    
//     // Display the user's message in the chatbox
//     const chatbox = document.getElementById('chatbox');
//     const userMessage = document.createElement('div');
//     userMessage.className = 'user-message';
//     userMessage.textContent = message;
//     chatbox.appendChild(userMessage);
    
//     // Clear the input field
//     messageInput.value = '';
    
//     // Send the message to the server and get the NLP response
//     fetch('/chat', {
//         method: 'POST',
//         headers: {
//             'Content-Type': 'application/json'
//         },
//         body: JSON.stringify({ query: message })
//     })
//     .then(response => response.json())
//     .then(data => {
//         // Display the chatbot's response in the chatbox
//         const nlpResponse = document.createElement('div');
//         nlpResponse.className = 'chatbot-message';
//         nlpResponse.textContent = data.results[0];
//         chatbox.appendChild(nlpResponse);
        
//         // Scroll to the bottom of the chatbox
//         chatbox.scrollTop = chatbox.scrollHeight;
//     })
//     .catch(error => {
//         console.error('Error:', error);
//     });
// });

function scrollToBottom() {
    const chatbox = document.getElementById('chatbox');

    // Ensure DOM has rendered the new content
    setTimeout(() => {
        chatbox.scrollTop = chatbox.scrollHeight; // Scrolls to the bottom
    }, 50); // Small delay ensures browser renders updates
}


document.getElementById('chatForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const messageInput = document.getElementById('message');
    const message = messageInput.value;
    
    // Display the user's message in the chatbox
    const chatbox = document.getElementById('chatbox');
    const userMessage = document.createElement('div');
    userMessage.className = 'user-message';
    userMessage.innerHTML = `<strong>User:</strong> ${message}`;
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
        // Display the chatbot's response in the chatbox
        const nlpResponse = document.createElement('div');
        nlpResponse.className = 'chatbot-message';
        nlpResponse.innerHTML = `<strong>NLP:</strong> ${data.results[0]}`;
        chatbox.appendChild(nlpResponse);
        
        // Scroll to the bottom of the chatbox
        // chatbox.scrollTop = chatbox.scrollHeight + 10;
        scrollToBottom();

    })
    .catch(error => {
        console.error('Error:', error);
    });
    // const chatboxC = document.querySelector('.chatbox-container');
    // const header = document.getElementById('chatbox-header');
    
    // let isDragging = false;
    // let offsetX = 0;
    // let offsetY = 0;
    
    // // Enable dragging
    // header.addEventListener('mousedown', (e) => {
    //     isDragging = true;
    //     offsetX = e.clientX - chatboxC.getBoundingClientRect().left;
    //     offsetY = e.clientY - chatboxC.getBoundingClientRect().top;
    
    //     chatboxC.style.cursor = 'grabbing';
    //     document.addEventListener('mousemove', onMouseMove);
    //     document.addEventListener('mouseup', onMouseUp);
    // });
    
    // function onMouseMove(e) {
    //     if (isDragging) {
    //         chatboxC.style.left = `${e.clientX - offsetX}px`;
    //         chatboxC.style.top = `${e.clientY - offsetY}px`;
    //     }
    // }
    
    // function onMouseUp() {
    //     isDragging = false;
    //     chatboxC.style.cursor = 'move';
    //     document.removeEventListener('mousemove', onMouseMove);
    //     document.removeEventListener('mouseup', onMouseUp);
    // }
    // window.addEventListener('resize', () => {
    //     const chatbox = document.querySelector('.chatbox-container');
    //     chatbox.style.maxWidth = `${window.innerWidth}px`;
    //     chatbox.style.maxHeight = `${window.innerHeight}px`;
    // });
    
});