let slideIndex = 1;
showSlides(slideIndex);

function plusSlides(n) {
  showSlides((slideIndex += n));
}

function currentSlide(n) {
  showSlides((slideIndex = n));
}

function showSlides(n) {
  let slides = document.getElementsByClassName("slide");
  let nextBtn = document.querySelector('.next');
  let prevBtn = document.querySelector('.prev');
  let counter = document.getElementById('counter'); // Add an element with id="counter" in your HTML

  if (n > slides.length) {
    slideIndex = 1;
  }
  if (n < 1) {
    slideIndex = slides.length;
  }

  for (let i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }

  slides[slideIndex - 1].style.display = "block";

  // Update image counter
  if (counter) {
    counter.textContent = `Image ${slideIndex} of ${slides.length}`;
  }

  // Disable/Enable buttons
  nextBtn.disabled = slideIndex === slides.length;
  prevBtn.disabled = slideIndex === 1;
}
