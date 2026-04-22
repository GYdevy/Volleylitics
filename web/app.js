import { renderCourtBackground, renderCourtRallies } from "./court.js";

const data = [
	{
		season: "2025-2026",
		matches: [
			{
				id: "s1m1",
				label: "אליצור אשקלון - גבעתיים",
				videoId: "dkr3LRiW5L0",
				photo: "./media/season-2025/final-banner.jpg",
				longestRally: "00:29",
				ralliesFile: "./data/matches/match17/rally_results.json",
				detailsFile: "./data/matches/match17/match_details.json"
			},
			{
				id: "s1m2",
				label: "Week 8: Falcons vs Storm",
				videoId: "dkr3LRiW5L0",
				photo: "./media/season-2025/week8.jpg",
				longestRally: "00:24",
				ralliesFile: "./data/matches/match18/rally_results.json"
			}
		]
	}
];
let currentRallies = [];
let activeSetFilter = "all";
let showAllTrajectories = false;
async function loadRalliesForMatch(matchObj) {
	const response = await fetch(matchObj.ralliesFile);
	if (!response.ok) {
		throw new Error(`Failed to load rallies: ${matchObj.ralliesFile}`);
	}
	return await response.json();
}
const browseToggleEl = document.getElementById("browseToggle");
const browseSidebarEl = document.getElementById("browseSidebar");
const sidebarBackdropEl = document.getElementById("sidebarBackdrop");
const browseCloseEl = document.getElementById("browseClose");
const seasonListEl = document.getElementById("seasonList");
const matchListEl = document.getElementById("matchList");
const matchesTitleEl = document.getElementById("matchesTitle");
const matchTitleEl = document.getElementById("matchTitle");
const matchVideoEl = document.getElementById("matchVideo");

const pointsEl = document.getElementById("heatmapPoints");
const setFilterEl = document.getElementById("setFilter");
const trajectoryToggleEl = document.getElementById("trajectoryToggle");
const statTotalEl = document.getElementById("statTotal");
const statLeftEl = document.getElementById("statLeft");
const statRightEl = document.getElementById("statRight");
const statLongestEl = document.getElementById("statLongest");
const heatmapSvgEl = document.getElementById("heatmap");
const matchDetailsEl = document.getElementById("matchDetails");
let currentSeason = data[0];
let currentMatch = currentSeason.matches[0];

let ytPlayer = null;
let ytApiReady = false;
let rallyStopTimeout = null;

function loadYouTubeAPI() {
	if (window.YT && window.YT.Player) {
		ytApiReady = true;
		createOrLoadPlayer();
		return;
	}

	const script = document.createElement("script");
	script.src = "https://www.youtube.com/iframe_api";
	document.head.appendChild(script);
}

window.onYouTubeIframeAPIReady = function() {
	ytApiReady = true;
	createOrLoadPlayer();
};

function createOrLoadPlayer() {
	if (!ytApiReady || !currentMatch.videoId) return;

	if (!document.getElementById("ytPlayer")) {
		matchVideoEl.innerHTML = '<div id="ytPlayer"></div>';
	}

	if (ytPlayer) {
		ytPlayer.loadVideoById(currentMatch.videoId);
		ytPlayer.pauseVideo();
		return;
	}

	ytPlayer = new YT.Player("ytPlayer", {
		width: "960",
		height: "540",
		videoId: currentMatch.videoId,
		playerVars: {
			playsinline: 1,
			rel: 0
		},
		events: {
			onReady: () => {
				ytPlayer.pauseVideo();
			}
		}
	});
}

function renderSeasons() {
	seasonListEl.innerHTML = "";

	data.forEach((seasonObj) => {
		const li = document.createElement("li");
		const button = document.createElement("button");

		button.className = `item ${seasonObj.season === currentSeason.season ? "active" : ""}`;
		button.textContent = seasonObj.season;

		button.addEventListener("click", async () => {
			currentSeason = seasonObj;
			currentMatch = seasonObj.matches[0];
			renderSeasons();
			renderMatches();
			await renderMatch();
			closeBrowseSidebar();

		});

		li.appendChild(button);
		seasonListEl.appendChild(li);
	});
}
function getFilteredRallies() {
	if (activeSetFilter === "all") {
		return currentRallies;
	}

	const setId = Number(activeSetFilter);
	return currentRallies.filter((rally) => rally.set_id === setId);
}

function refreshHeatmap() {
	const filteredRallies = getFilteredRallies();

	renderHeatmapPoints(filteredRallies);
	renderStats(filteredRallies, currentMatch.longestRally);
}
function renderMatches() {
	matchListEl.innerHTML = "";
	matchesTitleEl.textContent = `${currentSeason.season} Matches`;
	activeSetFilter = "all";
	showAllTrajectories = false;
	if (setFilterEl) setFilterEl.value = "all";
	if (trajectoryToggleEl) trajectoryToggleEl.checked = false;
	currentSeason.matches.forEach((matchObj) => {
		const li = document.createElement("li");
		const button = document.createElement("button");

		button.className = `item ${matchObj.id === currentMatch.id ? "active" : ""}`;
		button.textContent = matchObj.label;

		button.addEventListener("click", async () => {
			currentMatch = matchObj;
			renderMatches();
			await renderMatch();
			closeBrowseSidebar();

		});

		li.appendChild(button);
		matchListEl.appendChild(li);
	});
}
async function loadMatchDetails(matchObj) {
	const response = await fetch(matchObj.detailsFile);
	if (!response.ok) {
		throw new Error(`Failed to load match details: ${matchObj.detailsFile}`);
	}
	return await response.json();
}

function resetRallyPlaybackTimer() {
	if (rallyStopTimeout) {
		clearTimeout(rallyStopTimeout);
		rallyStopTimeout = null;
	}
}



function renderMatchDetails(details) {
	if (!details || !details.teams) {
		matchDetailsEl.innerHTML = "<p class='panel-help'>No match details available.</p>";
		return;
	}

	const rowsHtml = details.teams.map((team) => {
		const setCells = [0, 1, 2, 3, 4].map((i) => {
			const value = team.sets[i] ?? 0;
			return `<td class="td_ltr_center">${value}</td>`;
		}).join("");

		return `
			<tr>
				<td>${team.name}</td>
				${setCells}
				<td class="td_ltr_center">${team.total}</td>
			</tr>
		`;
	}).join("");

	matchDetailsEl.innerHTML = `
		<div class="table-wrap">
			<table class="common-table table result-by-set" id="match-table">
				<thead>
					<tr>
						<th>קבוצה/מערכה</th>
						<th class="td_ltr_center">1</th>
						<th class="td_ltr_center">2</th>
						<th class="td_ltr_center">3</th>
						<th class="td_ltr_center">4</th>
						<th class="td_ltr_center">5</th>
						<th class="td_ltr_center">סה"כ</th>
					</tr>
				</thead>
				<tbody>
					${rowsHtml}
				</tbody>
			</table>
		</div>
	`;
}
async function loadMatchData() {
	currentRallies = await loadRalliesForMatch(currentMatch);

	const details = await loadMatchDetails(currentMatch);
	renderMatchDetails(details);

	refreshHeatmap();
}

async function renderMatch() {
	matchTitleEl.textContent = currentMatch.label;

	createOrLoadPlayer();
	resetRallyPlaybackTimer();


	try {
		await loadMatchData();
	} catch (error) {
		console.error(error);
		pointsEl.innerHTML = "";
		matchDetailsEl.innerHTML = "<p class='panel-help'>Failed to load match details.</p>";
	}
}
function jumpToRally(startTime, endTime) {
	if (!ytPlayer) return;

	if (rallyStopTimeout) {
		clearTimeout(rallyStopTimeout);
		rallyStopTimeout = null;
	}

	ytPlayer.seekTo(startTime, true);
	ytPlayer.playVideo();

	if (typeof endTime === "number" && endTime > startTime) {
		const durationMs = (endTime - startTime + 1.5) * 1000;

		rallyStopTimeout = setTimeout(() => {
			if (ytPlayer) {
				ytPlayer.pauseVideo();
			}
		}, durationMs);
	}
}

function renderHeatmapPoints(rallies) {
	renderCourtRallies(
		pointsEl,
		rallies,
		(rally) => {
			jumpToRally(rally.start, rally.end);
		},
		{
			showAllTrajectories
		}
	);
}

function renderStats(rallies, longestRally) {
	const total = rallies.length;
	const left = rallies.filter((r) => r.landing_point && r.landing_point[0] < 4.5).length;
	const right = total - left;

	statTotalEl.textContent = String(total);
	statLeftEl.textContent = total ? `${Math.round((left / total) * 100)}%` : "0%";
	statRightEl.textContent = total ? `${Math.round((right / total) * 100)}%` : "0%";
	statLongestEl.textContent = longestRally;
}
function openBrowseSidebar() {
	browseSidebarEl.classList.add("open");
	sidebarBackdropEl.classList.add("open");
	browseToggleEl.setAttribute("aria-expanded", "true");
}

function closeBrowseSidebar() {
	browseSidebarEl.classList.remove("open");
	sidebarBackdropEl.classList.remove("open");
	browseToggleEl.setAttribute("aria-expanded", "false");
}


browseToggleEl.addEventListener("click", () => {
	const isOpen = browseSidebarEl.classList.contains("open");
	if (isOpen) {
		closeBrowseSidebar();
	} else {
		openBrowseSidebar();
	}
});

browseCloseEl.addEventListener("click", closeBrowseSidebar);
sidebarBackdropEl.addEventListener("click", closeBrowseSidebar);
setFilterEl.addEventListener("change", () => {
	activeSetFilter = setFilterEl.value;
	refreshHeatmap();
});

trajectoryToggleEl.addEventListener("change", () => {
	showAllTrajectories = trajectoryToggleEl.checked;
	refreshHeatmap();
});
renderCourtBackground(heatmapSvgEl);
renderSeasons();
renderMatches();
loadYouTubeAPI();
renderMatch();
