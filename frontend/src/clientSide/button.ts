import { addFriend, fetchUserInformation, init2fa, login, profile, register, setAvatar, verify2fa } from './user.js'
import { friendList } from './friends.js'
import { connected, handleConnection, navigate } from './front.js'
import { tdStop, TowerDefense } from './td.js'
import { editProfile } from './editInfoProfile.js'
import { displayNotification } from './notificationHandler.js'
import { Pong } from './pong.js'
import { displayTournaments, fetchTournamentBrackets, joinTournament, launchTournament } from './tournaments.js'
import { printMatchHistory } from './matchHistory.js'
import { TowerDefenseSpec } from './tdspec.js'
import { closeSSEConnection } from './serverSentEvent.js'
import { pongAgainstAi } from './invitation.js'

const mapButton: { [key: string]: () => void } = {
	'/connect': connectBtn,
	'/login': loginBtn,
	'/profile': profileBtn,
	'/logout': logoutBtn,
	'/factor': factor,
	'/towerMode': towerMode,
	'/towerRemote': towerRemote,
	'/pongMode': pongMode,
	'/pongRemote': pongRemote,
	'/pongLocal': pongLocal,
	'/pongWatch': pongWatch,
	'/towerWatch': towerWatch,
	'/tournaments': tournaments,
	'/lobby': lobby,
	'/matchHistory': matchHistory,
	'/pongVsia': pongAgainstAi,
	'/About': About,
	'/brackets': Brackets,
}

export function activateBtn(page: string) {
	if (page in mapButton)
		mapButton[page]()
}

function connectBtn() {
	document.getElementById('loginButton')?.addEventListener('click', (event: MouseEvent) => navigate('/login', event))
	const button = document.getElementById('registerButton') as HTMLButtonElement
	if (button)
		register(button)
}

function loginBtn() {
	document.getElementById('connectPageBtn')?.addEventListener('click', (event: MouseEvent) => navigate('/connect', event))
	const button = document.getElementById('loginButton') as HTMLButtonElement
	if (button)
		login(button)
}

async function profileBtn() {
	const avatarImg = document.getElementById('avatarProfile') as HTMLImageElement
	const avatar = sessionStorage.getItem('avatar')
	if (avatar)
		avatarImg.src = avatar
	await profile()
	await friendList()
	editProfile()
	document.getElementById('logout')?.addEventListener('click', (event: MouseEvent) => navigate('/logout', event))
	const addFriendBtn = document.getElementById('friendNameBtn') as HTMLButtonElement
	const addFriendIpt = document.getElementById('friendNameIpt') as HTMLButtonElement
	if (addFriendBtn && addFriendIpt)
		addFriendBtn.addEventListener('click', async () => {
			if (addFriendIpt.value.length >= 3) {
				await addFriend(addFriendIpt.value)
				await friendList()
			} else {
				displayNotification('Name of 3 characters minimum', { type: 'error' })
			}
		})
	const activeFA = sessionStorage.getItem('activeFA')
	if (activeFA) {
		document.getElementById('totalFactor')?.classList.add('hidden')
		document.getElementById('activeFactor')?.classList.remove('hidden')
	} else {
		const initFa = document.getElementById('initFa') as HTMLButtonElement
		if (initFa) {
			initFa.addEventListener('click', async () => {
				const qrcode = await init2fa()
				if (qrcode == undefined) {
					console.log('ErrorDisplay: qrcode not found!')
					return
				}
				const insertQrcode = document.getElementById('insertQrcode')
				if (insertQrcode) {
					const img = document.createElement('img')
					img.src = qrcode
					img.classList.add('h-3/4', 'w-3/4', 'p-4')
					insertQrcode.innerHTML = ''
					initFa.classList.add('hidden')
					insertQrcode.appendChild(img)
					const label = document.getElementById('codeFaInput')
					if (label)
						label.classList.remove('sr-only')
					const input = document.getElementById('inputVerify') as HTMLInputElement

					input.addEventListener('keydown', async (event: KeyboardEvent) => {
						if (event.key === 'Enter') {
							await verify2fa(input.value, 'You have enabled two-factor authentication.')
						}
					})
				}
			})
		}
	}
	document.getElementById('inputAvatar')?.addEventListener('change', async (event: Event) => {
		const target = event.target as HTMLInputElement
		await setAvatar(target)
	})
}

function logoutBtn() {
	handleConnection(false)
	const avatar = document.getElementById('avatar') as HTMLImageElement | null
	if (avatar) avatar.src = '../images/logout.png'
	const nickName = document.getElementById('nickName') as HTMLSpanElement | null
	if (nickName) nickName.textContent = ''
	closeSSEConnection()
}

function factor() {
	console.log('factor have been called !')
	document.getElementById('checkCode')?.addEventListener('click', async () => {
		const input = document.getElementById('inputVerify') as HTMLInputElement | null
		if (!input) {
			await navigate('/logout')
			displayNotification('Error from client, try again!', { type: 'error' })
			return
		}
		await verify2fa(input.value, 'The validity of your code has been confirmed')
	})
}

function pongMode() {
	document.getElementById('pongRemote')?.addEventListener('click', (event: MouseEvent) => navigate('/pongRemote', event))
	document.getElementById('pongLocal')?.addEventListener('click', (event: MouseEvent) => navigate('/pongLocal', event))
	document.getElementById('pongWatch')?.addEventListener('click', (event: MouseEvent) => navigate('/pongWatch', event))
	document.getElementById('pongVsia')?.addEventListener('click', (event: MouseEvent) => navigate('/pongVsia', event))
}

function towerMode() {
	document.getElementById('towerRemote')?.addEventListener('click', (event: MouseEvent) => navigate('/towerRemote', event))
	document.getElementById('towerWatch')?.addEventListener('click', (event: MouseEvent) => navigate('/towerWatch', event))
}

function towerRemote() {
	tdStop()
	TowerDefense()
}

function pongLocal() {
	Pong('local')
}

function pongRemote() {
	Pong('remote')
}

function pongWatch() {
	Pong('spec')
}

function towerWatch() {
	TowerDefenseSpec()
}

function tournaments() {
	const tournaments: { id: number, name: string } [] = [
		{ id: 4, name: 'Junior' },
		{ id: 8, name: 'Contender' },
		{ id: 16, name: 'Major' },
		{ id: 32, name: 'Worlds' }]
	for (const parse of tournaments)
		document.getElementById(`${parse.id}`)
			?.addEventListener('click', async (event) => {
				sessionStorage.setItem('idTournaments', JSON.stringify(parse.id))
				sessionStorage.setItem('nameTournaments', parse.name)
				await joinTournament(parse.id)
				await navigate('/lobby', event)
			})
}

async function lobby() {
	const id = sessionStorage.getItem('idTournaments')
	const name = sessionStorage.getItem('nameTournaments')
	if (!id || !name) {
		displayNotification('Please connect to an account.')
		await navigate('/home')
		return
	}
	const toIntId = Number.parseInt(id)
	if (isNaN(toIntId)) displayNotification('Invalid tournament ID.')
	await displayTournaments(toIntId, name)
	document.getElementById('launchTournamentBtn')?.addEventListener('click', async () => {
		await launchTournament()
	})
}

async function matchHistory() {
	await printMatchHistory()
}

function About() {
	const img = document.getElementById('imgToknow') as HTMLImageElement | null
	if (img) {
		if (connected) {
			const avatar = sessionStorage.getItem('avatar')
			if (avatar) img.src = avatar
			else img.src = '../images/login.png'
		} else img.src = '../images/logout.png'
	}
}

async function Brackets() {
	try {
		const idTournament = sessionStorage.getItem('idTournaments')
		if (!idTournament) throw new Error('No ID Tournament found!')
		const bracketsData = await fetchTournamentBrackets(Number(idTournament))
		console.log('BRACKETS DATA', bracketsData)
		if (!bracketsData) throw new Error('No brackets data')

		const template = document.getElementById('bracketsTmp') as HTMLTemplateElement | null
		const list = document.getElementById('playerList') as HTMLUListElement | null

		if (!template || !list) throw new Error('Missing template or list')

		for (const bracket of bracketsData) {
			const playerId: number[] = []
			if (bracket.id1 !== 0) playerId.push(bracket.id1)
			if (bracket.id2 !== 0) playerId.push(bracket.id2)
			const userData = await fetchUserInformation(playerId)
			if (!userData) continue

			const clone = template.content.cloneNode(true) as DocumentFragment

			const playerOneImg = clone.querySelector('.player-one-img') as HTMLImageElement | null
			const playerOneName = clone.querySelector('.player-one-name') as HTMLElement | null
			const playerTwoImg = clone.querySelector('.player-two-img') as HTMLImageElement | null
			const playerTwoName = clone.querySelector('.player-two-name') as HTMLElement | null
			if (playerOneImg && userData[0]?.avatar) playerOneImg.src = userData[0].avatar
			if (playerOneName) playerOneName.innerText = userData[0]?.nickName
			if (playerTwoImg && userData[1]?.avatar) playerTwoImg.src = userData[1].avatar
			if (playerTwoName) playerTwoName.innerText = userData[1]?.nickName
			list.appendChild(clone)
		}
	} catch (error) {
		console.log('Brackets Error: ', error)
		displayNotification('Can\'t show phase of tournament', { type: 'error' })
	}
}
