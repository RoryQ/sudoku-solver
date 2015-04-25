log = (status) ->
    $("#status").html status
$ ->
    $("#upload_button").click ->
        $("input[type=file]").click()
    form = $("#upload_form")
    form.fileupload
        autoUpload: true
        dataType: "xml"
        add: (event, data) ->
            log "fetching params"
            $.get("{{ url_for('params') }}").done (params) ->
                form.find('input[name=key]').val(params.key)
                form.find('input[name=policy]').val(params.policy)
                form.find('input[name=signature]').val(params.signature)
                form.find('input[name=success_action_redirect]').val(params.success_action_redirect)
                data.submit()
        send: (event, data) ->
            log "sending"
        progress: (event, data) ->
            $("#progress_bar").css "width", "#{Math.round((event.loaded / event.total) * 1000) / 10}%"
        fail: (event, data) ->
            log "failure"
        success: (event, data) ->
            log "success"
        done: (event, data) ->
            log "done"