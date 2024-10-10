import streamlit as st
from PIL import Image, ImageDraw
from utils.sidebar import display_copyright

display_copyright()

def make_circle(image):
    """Creates a circular mask for the given image."""
    width, height = image.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, width, height), fill=255)

    circular_image = Image.new("RGBA", (width, height))
    circular_image.paste(image, (0, 0), mask)
    return circular_image


def display_profile(image_path, name, roll, mobile, email, linkedIn, qualification, github=None):
    """Displays a developer's profile with an image and contact information."""
    image = Image.open(image_path)
    circular_image = make_circle(image.resize((230, 260)))
    st.image(circular_image)
    st.subheader(name)
    st.write(f"- MBA at VIT University - {roll}")

    # --- EXPERIENCE & QUALIFICATIONS ---
    st.subheader("Qualifications")
    st.write(f"- {qualification}")

    st.subheader("Contact Information")
    st.write(
        f""" 
        - **Mobile:** {mobile}
        - **Email:** [{email}](mailto:{email})
        - **LinkedIn:** [{linkedIn}](https://www.linkedin.com/in/{linkedIn}/)
        """
    )
    if github:
        st.write(f"- **GitHub:** [github.com/{github}](https://github.com/{github})")

# Main content
st.markdown("<h3 style='text-align: center;'>About the Developers</h3>", unsafe_allow_html=True)

# Create two columns for the profiles
col1, col2, col3 = st.columns(3, gap="large")
with col1:
    display_profile(
        'images/pic_4.jpeg',
        "Ananya S",
        "23MBA0012",
        "+91-6369489092",
        "ananyasid111@gmail.com",
        "ananya-siddharth",
        "Bachelor of Hons Agriculture"
    )

with col2:
    display_profile(
        'images/pic_2.jpg',
        "Sandeep M",
        "23MBA0015",
        "+91-9025682538",
        "sandeepmurugesan16@gmail.com",
        "sandeepmurugesan",
        "Bachelor of Commerce"
    )

with col3:
    display_profile(
        'images/pic_3.jpg',
        "Naveen AS",
        "23MBA0063",
        "+91-9092486777",
        "asnaveen21@gmail.com",
        "naveen-as",
        "Mechanical Engineering"
    )

st.markdown("---")

col4, col5 = st.columns(2, gap="large")


with col4:
    display_profile(
        'images/pic_1.png',
        "Jayaprakash S",
        "23MBA0103",
        "+91-8754813384",
        "prakash4830jp@gmail.com",
        "jayaprakash-s",
        "Computer Science and Engineering",
        "prakash4830"  # Adding GitHub account only for Jayaprakash
    )

with col5:
    display_profile(
        'images/pic_5.jpeg',
        "Manoj S",
        "23MBA0131",
        "+91-7397791923",
        "manoshekar9@gmail.com",
        "manoj-kumaar",
        "Bachelor of Hons Agriculture"
    )


# Your page content here

# At the end of the page


# Additional profile

