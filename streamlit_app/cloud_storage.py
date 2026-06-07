import cloudinary
import cloudinary.uploader
import streamlit as st


def configure_cloudinary():

    cloudinary.config(
        cloud_name=st.secrets["CLOUDINARY_CLOUD_NAME"],
        api_key=st.secrets["CLOUDINARY_API_KEY"],
        api_secret=st.secrets["CLOUDINARY_API_SECRET"],
        secure=True
    )


def upload_image_to_cloudinary(image_path):

    configure_cloudinary()

    result = cloudinary.uploader.upload(
        image_path,
        folder="skin_ai_demo"
    )

    return {
        "secure_url": result["secure_url"],
        "public_id": result["public_id"]
    }


def delete_image_from_cloudinary(public_id):

    configure_cloudinary()

    if public_id:
        cloudinary.uploader.destroy(public_id)