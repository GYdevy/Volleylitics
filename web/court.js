const SVG_NS = "http://www.w3.org/2000/svg";

// =========================
// SINGLE SOURCE OF TRUTH
// =========================

export const COURT = {
	svgWidth: 600,
	svgHeight: 765,

	outerColor: "#508250",
	courtColor: "#DC8C3C",
	lineColor: "#FFFFFF",

	courtWidth: 306,

	metersX: 9,
	metersY: 18,
};

COURT.courtHeight = COURT.courtWidth * (COURT.metersY / COURT.metersX);
COURT.courtX = (COURT.svgWidth - COURT.courtWidth) / 2;
COURT.courtY = (COURT.svgHeight - COURT.courtHeight) / 2;
COURT.netY = COURT.courtY + COURT.courtHeight / 2;
COURT.centerX = COURT.courtX + COURT.courtWidth / 2;
COURT.threeMeterOffset = (3 / 9) * (COURT.courtHeight / 2);

COURT.lowerLeft = COURT.courtX;
COURT.lowerTop = COURT.netY;
COURT.lowerWidth = COURT.courtWidth;
COURT.lowerHeight = COURT.courtHeight / 2;
const SET_COLORS = {
	1: "#ff4d4f",
	2: "#52c41a",
	3: "#1677ff",
	4: "#faad14",
	5: "#b37feb",
	default: "#d9d9d9",
};

function createSvgElement(tag, attrs = {}) {
	const el = document.createElementNS(SVG_NS, tag);
	for (const [key, value] of Object.entries(attrs)) {
		el.setAttribute(key, String(value));
	}
	return el;
}

export function getSetColor(setId) {
	return SET_COLORS[setId] || SET_COLORS.default;
}

export function courtToSvg(x, y) {
	return {
		x: COURT.lowerLeft + (x / 9) * COURT.lowerWidth,
		y: COURT.lowerTop + ((9 - y) / 9) * COURT.lowerHeight,
	};
}

// =========================
// COURT BACKGROUND
// =========================
export function renderCourtBackground(svgEl) {
	if (!svgEl) return;

	svgEl.setAttribute("viewBox", `0 0 ${COURT.svgWidth} ${COURT.svgHeight}`);

	const courtGroup = svgEl.querySelector("#heatmapCourt");
	if (!courtGroup) return;

	courtGroup.innerHTML = "";

	// outer area
	courtGroup.appendChild(
		createSvgElement("rect", {
			x: 0,
			y: 0,
			width: COURT.svgWidth,
			height: COURT.svgHeight,
			fill: COURT.outerColor,
		})
	);

	// court fill
	courtGroup.appendChild(
		createSvgElement("rect", {
			x: COURT.courtX,
			y: COURT.courtY,
			width: COURT.courtWidth,
			height: COURT.courtHeight,
			fill: COURT.courtColor,
		})
	);

	// boundary
	courtGroup.appendChild(
		createSvgElement("rect", {
			x: COURT.courtX,
			y: COURT.courtY,
			width: COURT.courtWidth,
			height: COURT.courtHeight,
			fill: "transparent",
			stroke: COURT.lineColor,
			"stroke-width": 2,
		})
	);

	// net
	courtGroup.appendChild(
		createSvgElement("line", {
			x1: COURT.courtX,
			y1: COURT.netY,
			x2: COURT.courtX + COURT.courtWidth,
			y2: COURT.netY,
			stroke: COURT.lineColor,
			"stroke-width": 3,
		})
	);

	// 3m lines
	for (const y of [COURT.netY - COURT.threeMeterOffset, COURT.netY + COURT.threeMeterOffset]) {
		courtGroup.appendChild(
			createSvgElement("line", {
				x1: COURT.courtX,
				y1: y,
				x2: COURT.courtX + COURT.courtWidth,
				y2: y,
				stroke: COURT.lineColor,
				"stroke-width": 2,
			})
		);
	}

	// center vertical line
	courtGroup.appendChild(
		createSvgElement("line", {
			x1: COURT.centerX,
			y1: COURT.courtY,
			x2: COURT.centerX,
			y2: COURT.courtY + COURT.courtHeight,
			stroke: "#838383",
			"stroke-width": 2,
			"stroke-dasharray": "10 10",
		})
	);
}

// =========================
// OVERLAY
// =========================
export function clearCourtOverlay(pointsEl) {
	pointsEl.innerHTML = "";
}

export function drawAttackLine(pointsEl, rally) {
	if (!rally.attack_point || !rally.landing_point) return null;

	const [ax, ay] = rally.attack_point;
	const [lx, ly] = rally.landing_point;

	const a = courtToSvg(ax, ay);
	const l = courtToSvg(lx, ly);

	const color = getSetColor(rally.set_id);

	const line = createSvgElement("line", {
		x1: a.x,
		y1: a.y,
		x2: l.x,
		y2: l.y,
		stroke: color,
		"stroke-width": 2.5,
		"stroke-linecap": "round",
		opacity: 0.9,
	});

	pointsEl.appendChild(line);
	return line;
}

export function enableHeatmapZoom(svgEl) {
	if (!svgEl) return;

	if (!svgEl.hasAttribute("viewBox")) {
		svgEl.setAttribute("viewBox", "0 0 700 760");
	}

	const initial = svgEl.getAttribute("viewBox").split(/\s+/).map(Number);

	const defaultViewBox = {
		x: initial[0],
		y: initial[1],
		width: initial[2],
		height: initial[3]
	};

	let viewBox = { ...defaultViewBox };
	let isPanning = false;
	let panStart = null;
	let panStartViewBox = null;

	function applyViewBox() {
		svgEl.setAttribute(
			"viewBox",
			`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`
		);
	}

	svgEl.addEventListener("wheel", (e) => {
		e.preventDefault();

		const rect = svgEl.getBoundingClientRect();
		const mx = (e.clientX - rect.left) / rect.width;
		const my = (e.clientY - rect.top) / rect.height;

		const scale = e.deltaY > 0 ? 1.1 : 0.9;

		const newWidth = viewBox.width * scale;
		const newHeight = viewBox.height * scale;

		viewBox.x += (viewBox.width - newWidth) * mx;
		viewBox.y += (viewBox.height - newHeight) * my;
		viewBox.width = newWidth;
		viewBox.height = newHeight;

		applyViewBox();
	}, { passive: false });

	svgEl.addEventListener("mousedown", (e) => {
		isPanning = true;
		panStart = { x: e.clientX, y: e.clientY };
		panStartViewBox = { ...viewBox };
		svgEl.style.cursor = "grabbing";
		e.preventDefault();
	});

	window.addEventListener("mousemove", (e) => {
		if (!isPanning) return;

		const rect = svgEl.getBoundingClientRect();

		const dxPx = e.clientX - panStart.x;
		const dyPx = e.clientY - panStart.y;

		const dx = (dxPx / rect.width) * panStartViewBox.width;
		const dy = (dyPx / rect.height) * panStartViewBox.height;

		viewBox.x = panStartViewBox.x - dx;
		viewBox.y = panStartViewBox.y - dy;

		applyViewBox();
	});

	window.addEventListener("mouseup", () => {
		isPanning = false;
		svgEl.style.cursor = "grab";
	});

	svgEl.addEventListener("mouseleave", () => {
		isPanning = false;
		svgEl.style.cursor = "grab";
	});

	svgEl.addEventListener("dblclick", () => {
		viewBox = { ...defaultViewBox };
		applyViewBox();
	});

	svgEl.style.cursor = "grab";
	applyViewBox();

	return function resetHeatmapZoom() {
		viewBox = { ...defaultViewBox };
		applyViewBox();
	};
}
export function drawLandingPoint(pointsEl, rally, onClick) {
	if (!rally.landing_point) return null;

	const [lx, ly] = rally.landing_point;
	const p = courtToSvg(lx, ly);
	const color = getSetColor(rally.set_id);

	const circle = createSvgElement("circle", {
		cx: p.x,
		cy: p.y,
		r: 7,
		fill: color,
		class: "point",
		tabindex: 0,
	});

	circle.setAttribute(
		"aria-label",
		`Rally ${rally.rally_id}, set ${rally.set_id}, jump to ${Math.round(rally.start)} seconds`
	);

	if (onClick) {
		circle.addEventListener("click", () => onClick(rally));
		circle.addEventListener("keydown", (e) => {
			if (e.key === "Enter" || e.key === " ") {
				e.preventDefault();
				onClick(rally);
			}
		});
	}

	pointsEl.appendChild(circle);
	return circle;
}

export function drawRallyLabel(pointsEl, rally) {
	if (!rally.landing_point) return;

	const [lx, ly] = rally.landing_point;
	const p = courtToSvg(lx, ly);

	const textOutline = createSvgElement("text", {
		x: p.x + 10,
		y: p.y - 10,
		fill: "white",
		"font-size": 12,
		"font-weight": "700",
		"paint-order": "stroke",
		stroke: "white",
		"stroke-width": 3,
	});
	textOutline.textContent = String(rally.rally_id);

	const text = createSvgElement("text", {
		x: p.x + 10,
		y: p.y - 10,
		fill: "black",
		"font-size": 12,
		"font-weight": "700",
	});
	text.textContent = String(rally.rally_id);

	pointsEl.appendChild(textOutline);
	pointsEl.appendChild(text);
}

export function drawRally(pointsEl, rally, onClick, options = {}) {
	const { showAllTrajectories = false } = options;

	let lineEl = null;

	if (showAllTrajectories) {
		lineEl = drawAttackLine(pointsEl, rally);
	}

	const circleEl = drawLandingPoint(pointsEl, rally, onClick);
	drawRallyLabel(pointsEl, rally);

	if (!showAllTrajectories && circleEl) {
		circleEl.addEventListener("mouseenter", () => {
			if (!lineEl) lineEl = drawAttackLine(pointsEl, rally);
		});

		circleEl.addEventListener("mouseleave", () => {
			if (lineEl) {
				lineEl.remove();
				lineEl = null;
			}
		});
	}
}

export function renderCourtRallies(pointsEl, rallies, onClick, options = {}) {
	clearCourtOverlay(pointsEl);

	rallies.forEach((rally) => {
		drawRally(pointsEl, rally, onClick, options);
	});
}
