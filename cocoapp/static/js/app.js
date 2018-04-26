$(document).ready(function() {
    var $status = $('.status');

    $('#img').change(function(event) {
        var obj = $(this)[0];

        $status.html('');

        if (obj.files && obj.files[0]) {
            var fileReader = new FileReader();
            fileReader.onload = function(event) {
                $('.img-area').html(
                    `<img class='loaded-img' src='${event.target.result}'/>`
                );
            }
            fileReader.readAsDataURL(obj.files[0]);
        }
    });

    $('form').submit(function(event) {
        event.preventDefault();

        if ($('#img')[0].files.length === 0) {
            return false;
        }

        var imageData = new FormData($(this)[0]);

        $status.html(
            `<span class='eval'>Evaluating...</span>`
        );

        $.ajax({
            url: '/predict',
            type: 'POST',
            processData: false,
            contentType: false,
            dataType: 'json',
            data: imageData,

            success: function(responseData) {
                console.log(responseData);
                if (responseData.error === 'bad-type') {
                    $status.html(
                        `<span class='eval'>Valid file types are .jpg and .png</span>`
                    );
                } else {
                    let data = JSON.stringify(responseData, null, '\t');

                    $status.html(
                        `<span class='result success'>Results</span>
                         <pre>${data}</pre>`
                    );
                }
            },
            error: function() {
                $status.html(
                    `<span class='eval'>Something went wrong, try again later.</span>`
                );
            }
        });
    });

});