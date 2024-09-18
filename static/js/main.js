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
