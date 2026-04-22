const stepButtons = Array.from(document.querySelectorAll(".pipeline-step"));
const stepSections = Array.from(document.querySelectorAll(".how-step-section"));

function setActiveStep(id) {
	stepButtons.forEach((btn) => {
		btn.classList.toggle("active", btn.dataset.target === id);
	});
}

stepButtons.forEach((btn) => {
	btn.addEventListener("click", () => {
		const targetId = btn.dataset.target;
		const section = document.getElementById(targetId);
		if (!section) return;

		section.scrollIntoView({
			behavior: "smooth",
			block: "start"
		});

		setActiveStep(targetId);
	});
});

const observer = new IntersectionObserver(
	(entries) => {
		const visible = entries
			.filter((entry) => entry.isIntersecting)
			.sort((a, b) => b.intersectionRatio - a.intersectionRatio);

		if (visible.length > 0) {
			setActiveStep(visible[0].target.id);
		}
	},
	{
		rootMargin: "-20% 0px -60% 0px",
		threshold: [0.2, 0.4, 0.6]
	}
);

stepSections.forEach((section) => observer.observe(section));
