import { Chart, DoughnutController, ArcElement, Tooltip, Legend, ChartConfiguration } from 'chart.js';

Chart.register(DoughnutController, ArcElement, Tooltip, Legend);

interface ChartData {
	wins: number;
	losses: number;
	draws: number;
	mmr: number;
	canvasId: string;
	mmrElementId: string;
}

// Crée un donut chart
function createDonutChart({ wins, losses, draws, mmr, canvasId, mmrElementId }: ChartData) {
	const ctx = document.getElementById(canvasId) as HTMLCanvasElement;
	const mmrElement = document.getElementById(mmrElementId);
	if (!ctx || !mmrElement) return;

	mmrElement.textContent = mmr.toString();

	const config: ChartConfiguration<'doughnut'> = {
		type: 'doughnut',
		data: {
			labels: ['Victoires', 'Défaites', 'Égalités'],
			datasets: [{
				data: [wins, losses, draws],
				backgroundColor: ['#22c55e', '#ef4444', '#eab308'],
				cutout: '70%',
			}]
		},
		options: {
			responsive: true,
			plugins: {
				legend: {
					labels: {
						color: '#ffffff'
					}
				}
			}
		}
	};

	new Chart(ctx, config);
}

// Données pour les deux jeux
const pongGameData: ChartData = {
	wins: 30,
	losses: 15,
	draws: 5,
	mmr: 1200,
	canvasId: 'donutChartPong',
	mmrElementId: 'mmrPong'
};

const towerDefenseData: ChartData = {
	wins: 20,
	losses: 10,
	draws: 10,
	mmr: 950,
	canvasId: 'donutChartTd',
	mmrElementId: 'mmrTD'
};

export function drawChart(data: ChartData) {
	createDonutChart(pongGameData);
	createDonutChart(towerDefenseData);
}