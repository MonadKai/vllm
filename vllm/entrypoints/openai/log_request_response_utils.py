__all__ = ["serialize_request_without_media"]


FAKE_MEDIA_CONTENT_PLACEHOLDER = "[FAKE_MEDIA_CONTENT]"


def serialize_request_without_media(request):
    data = request.model_dump()

    if "messages" in data:
        for message in data["messages"]:
            if "content" in message and isinstance(message["content"], list):
                filtered_content = []
                for content_part in message["content"]:
                    if isinstance(content_part, dict):
                        content_type = content_part.get("type", "")
                        if content_type not in ["audio_url", "image_url", "video_url"]:
                            filtered_content.append(content_part)
                        else:
                            filtered_content.append(
                                {"type": content_type, "url": FAKE_MEDIA_CONTENT_PLACEHOLDER}
                            )
                    else:
                        filtered_content.append(content_part)
                message["content"] = filtered_content

    return data
