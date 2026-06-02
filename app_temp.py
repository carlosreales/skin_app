from cloud_storage import upload_image_to_cloudinary

url = upload_image_to_cloudinary(
    "uploads/20260602_011319.jpg"
)

print(url)