$(document).ready(function() {
    $('#question-form').on('submit', function(event) {
        event.preventDefault();
        var question = $('#question').val();
        $.ajax({
            url: '/search',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ query: question }),
            success: function(response) {
                var answer = response.results[0];
                $('#chat-content').append('<p><strong>You:</strong> ' + question + '</p>');
                $('#chat-content').append('<p><strong>Bot:</strong> ' + answer + '</p>');
                $('#question').val('');
            }
        });
    });
});
document.addEventListener("DOMContentLoaded", function () {
    const chatbox = document.querySelector(".chatbox-container");

    // Load saved size if available
    const savedWidth = localStorage.getItem("chatbox-width");
    const savedHeight = localStorage.getItem("chatbox-height");

    if (savedWidth && savedHeight) {
        chatbox.style.width = savedWidth;
        chatbox.style.height = savedHeight;
    }

    // Save size on resize
    let resizeTimeout;
    chatbox.addEventListener("mouseup", () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            localStorage.setItem("chatbox-width", chatbox.style.width);
            localStorage.setItem("chatbox-height", chatbox.style.height);
        }, 500); // Save after resizing stops
    });
});
