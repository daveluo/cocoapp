$(document).ready(function() {
    var $status = $('.status');
    var results = {
        bboxes: []
    };

   var drawBBox = function() {
        var threshold = parseInt($("#bbox-slider").text());
        var c = document.getElementById("img-canvas");
        var ctx = c.getContext("2d");
        var img = document.getElementById("loaded-img");

        let height = 500;
        let width = 500;
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(img,0,0, width, height);

        results.bboxes.forEach(function(d) {
            var x1, y1, x2, y2;
            if (d.score > threshold) {
                ctx.beginPath();
                x1 = parseInt(d.x1 * width);
                y1 = parseInt(d.y1 * height);
                x2 = parseInt(d.x2 * width);
                y2 = parseInt(d.y2 * height);
                ctx.rect(x1, y1, x2, y2);
                ctx.lineWidth = 3;
                ctx.strokeStyle = 'white';
                ctx.stroke();

            }
        });
    };

    // With JQuery
    $('#bbox-slider').slider({
        formatter: function(value) {
            return value;
        }
    })
    .on('slide', drawBBox)
    .data('slider');

    $('#img').change(function(event) {
        var obj = $(this)[0];

        $status.html('');

        if (obj.files && obj.files[0]) {
            var fileReader = new FileReader();
            fileReader.onload = function(event) {
                $('.img-hidden').html(
                    `<img id='loaded-img' src='${event.target.result}'/>`
                );
                var c = document.getElementById("img-canvas");
                var ctx = c.getContext("2d");
                var img = document.getElementById("loaded-img");
                img.addEventListener("load", function(e) {
                ctx.drawImage(img,0,0, 500, 500);
                });
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
                if (responseData.error === 'bad-type') {
                    $status.html(
                        `<span class='eval'>Valid file types are .jpg and .png</span>`
                    );
                } else {
                    results["bboxes"] = responseData["bboxes"];
                    let preData = JSON.stringify(responseData, null, '\t');
                    $status.html(
                        `<span class='result success'>Results</span>
                         <pre>${preData}</pre>`
                    );
                    // Draw Bounding boxes
                    drawBBox();
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