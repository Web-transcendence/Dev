import {addFriend, fetchUserInformation, removeFriend, UserData} from './user.js'
import {Pong} from './pong.js'
import {displayNotification, hideNotification} from './notificationHandler.js'
import {navigate} from './front.js'
import {loadPart} from './insert.js'
import {openModal} from './modal.js'
import {TowerDefense} from './td.js'

let abortController: AbortController | null = null

const parseSSEMessage = (raw: string): { event: string, stringData: string } => {
	const result: Record<string, string> = {}
	const lines = raw.split('\n')

	for (const line of lines) {
		const [key, ...rest] = line.split(':')
		result[key.trim()] = rest.join(':').trim()
	}
	return {event: result.event, stringData: result.data}
}

export async function sseConnection() {
	try {
		const tempController = new AbortController()
		const token = sessionStorage.getItem('token')
		const res = await fetch(`/user-management/sse`, {
			method: 'GET',
			headers: {
				'Content-Type': 'text/event-stream',
				'authorization': 'Bearer ' + token,
			},
			signal: tempController.signal,
		})
		if (res.status === 100)
			return
		if (!res.ok) {
			const error = await res.json()
			console.log(error)
			//notifyhandling
			return
		}

		abortController = tempController

		console.log('sse connection')
		const reader = res.body?.pipeThrough(new TextDecoderStream()).getReader() ?? null
		while (reader) {
			const {value, done} = await reader.read()
			if (done) break
			if (value.startsWith('retry: ')) continue
			const parse = parseSSEMessage(value)
			if (parse.event in mapEvent)
				mapEvent[parse.event](JSON.parse(parse.stringData))
		}
	} catch (err) {
		if (err.name === 'AbortError')
			console.log('sse connection aborted')
		else
			console.error(err)
	}
}

export function closeSSEConnection() {
	if (abortController) {
		abortController.abort()
		console.log('SSE closed')
		abortController = null
	}
}

export const CreateFriendLi = async (id: number, key: string, tmpName: string) => {
	console.log(`this id ${id} have to be add in the friendlist`)
	const list = document.getElementById(key)
	const [userData] = await fetchUserInformation([id])
	const template = document.getElementById(tmpName) as HTMLTemplateElement | null

	if (list && template) {
		const clone = template.content.cloneNode(true) as HTMLElement
		const item = clone.querySelector('li')
		if (item) item.id = `friendId-${userData.id}`
		const img = clone.querySelector('img')
		if (img && userData.avatar) img.src = userData.avatar
		const name = clone.querySelector('.name')
		if (name) name.textContent = userData.nickName
		console.log('logOnline :', userData.online)
		if (userData.online && key === 'acceptedList') {
			clone.querySelector('.online')?.classList.remove('hidden')
			clone.querySelector('.inviteFriend')?.classList.remove('hidden')
		}
		if (key === 'receivedList') {
			clone.querySelector('.accept-btn')?.addEventListener('click', () => {
				addFriend(userData.nickName)
				document.getElementById(`friendId-${id}`)?.remove()
				hideNotification(0, id)
				CreateFriendLi(id, 'acceptedList', 'acceptedTemplate')
			})
			clone.querySelector('.decline-btn')?.addEventListener('click', () => {
				removeFriend(userData.nickName)
				document.getElementById(`friendId-${id}`)?.remove()
				hideNotification(0, id)
			})
		}
		if (key === 'acceptedList')
			clone.querySelector('.inviteFriend')?.addEventListener('click', async () => openModal(userData.nickName, userData.id))
		list.appendChild(clone)
	}
}


const notifyNewFriend = async ({id}: { id: number }) => {
	console.log(`this id ${id} have to be add in the friendlist`)
	document.getElementById(`friendId-${id}`)?.remove()
	await CreateFriendLi(id, 'acceptedList', 'acceptedTemplate')
}

const notifyFriendInvitation = async ({id}: { id: number }) => {
	console.log(`this friend has to be added in the invitation list`)
	await CreateFriendLi(id, 'receivedList', 'receivedTemplate')

	const [userData] = await fetchUserInformation([id])
	displayNotification('New  friend request!', {
		type: 'invitation',
		onAccept: async () => {
			if (await addFriend(userData.nickName)) {
				await CreateFriendLi(userData.id, 'acceptedList', 'acceptedTemplate')
			}
		},
		onRefuse: async () => {
			if (await removeFriend(userData.nickName)) {
				document.getElementById(`friendId-${userData.id}`)?.remove()
			}
		},

	}, userData)
}

const notifyFriendRemoved = ({id}: { id: number }) => {
	console.log(`this friend have to be supressed from the friendlist`)
	const list = document.getElementById(`friendId-${id}`)
	if (list) list.remove()
}

const notifyDisconnection = ({id}: { id: number }) => {
	console.log(`this friend have to be marked as unconnected`)
	const list = document.getElementById(`friendId-${id}`)
	if (list) {
		list.querySelector('.online')?.classList.add('hidden')
		list.querySelector('.inviteFriend')?.classList.add('hidden')
	}
}

const notifyConnection = ({id}: { id: number }) => {
	console.log(`this friend have to be marked as connected`)
	const list = document.getElementById(`friendId-${id}`)
	if (list) {
		list.querySelector('.online')?.classList.remove('hidden')
		list.querySelector('.inviteFriend')?.classList.remove('hidden')
	}
}

const notifyJoinTournament = async ({id, maxPlayer}: { id: number, maxPlayer: number }) => {
	console.log(`the user with the id ${id} joined my tournament`)
	const playerList = document.getElementById('playerList')
	const playerTmp = document.getElementById('playerTemplate') as HTMLTemplateElement | null
	const [{nickName, avatar}]: UserData[] = await fetchUserInformation([id])
	if (!playerList || !playerTmp) {
		displayNotification(`Error Can't find Tournaments`)
		await navigate('/home')
		return
	}
	const clone = playerTmp.content.cloneNode(true) as HTMLElement | null
	if (!clone) {
		displayNotification('Error 1.2 occur, please refresh your page.')
		return
	}
	const item = clone.querySelector('li')
	if (!item) {
		displayNotification('Error 2 occur, please refresh your page.')
		return
	}
	item.id = `itemId-${id}`
	const span = item.querySelector('span')
	if (span) {
		span.id = `spanId-${id}`
		span.innerText = nickName
	}
	const img = item.querySelector('img')
	if (img) {
		img.id = `imgId-${id}`
		if (avatar) img.src = avatar
		else img.src = '../images/login.png'
	}
	playerList.appendChild(item)
	const numberOfPlayer = document.getElementById(`numberOfPlayer`)
	if (numberOfPlayer) {
		const players = document.querySelectorAll('#playerList li')
		const number = players.length
		numberOfPlayer.innerText = `${number}/${maxPlayer}`
	}
}

const notifyQuitTournament = ({id, maxPlayer}: { id: number, maxPlayer: number }) => {
	console.log(`the user with the id ${id} Quit my tournament`)
	const list = document.getElementById(`itemId-${id}`)
	if (list) list.remove()
	const numberOfPlayer = document.getElementById(`numberOfPlayer`)
	if (numberOfPlayer) {
		const players = document.querySelectorAll('#playerList li')
		const number = players.length
		numberOfPlayer.innerText = `${number}/${maxPlayer}`
	}
}

const notifyInvitationPong = async ({roomId, id}: { roomId: number, id: number }) => {
	const [userData] = await fetchUserInformation([id])
	displayNotification('Invitation to play Pong', {
		type: 'invitation',
		onAccept: async () => {
			await navigate('/pongFriend')
			Pong('remote', roomId)
			console.log('Accepted invite')
		},
		onRefuse: async () => {
			console.log('Close invite because refused')
		},
	}, userData)
}

const notifyInvitationTowerDefense = async ({roomId, id}: { roomId: number, id: number }) => {
	const [userData] = await fetchUserInformation([id])
	displayNotification('Invitation to Play Tower-Defense', {
		type: 'invitation',
		onAccept: async () => {
			await loadPart('/towerFriend')
			TowerDefense(roomId)
			console.log('Accepted invite')
		},
		onRefuse: async () => {
			console.log('Close invite because refused')
		},
	}, userData)
}

const winBracket = async ({id}: { id: number }) => {
	displayNotification('You win the game the tournament continue !')
	console.log('You win the game the tournament continue !')
	await loadPart('/brackets')
}

const winTournament = async () => {
	displayNotification(`Congratulations, you win the tournament !!!`)
	console.log(`Congratulations, you win the tournament !!!`)
	sessionStorage.removeItem('idTournaments')
	sessionStorage.removeItem('nameTournaments')
	await loadPart('/home')
}

const loseTournament = async ({id}: { id: number }) => {
	displayNotification('You have lost the tournament :/', {type: 'error'})
	console.log('You have lost the tournament :/')
	sessionStorage.removeItem('idTournaments')
	sessionStorage.removeItem('nameTournaments')
	await loadPart('/home')
}


const notifyInvitationTournamentPong = async ({roomId}: { roomId: number }) => {
	console.log('RoomId', roomId)
	await loadPart('/pongTournament')
	Pong('remote', roomId)
}

const mapEvent: { [key: string]: (data: any) => void } = {
	'joinTournament': notifyJoinTournament,
	'quitTournament': notifyQuitTournament,
	'newFriend': notifyNewFriend,
	'friendInvitation': notifyFriendInvitation,
	'friendRemoved': notifyFriendRemoved,
	'connection': notifyConnection,
	'disconnection': notifyDisconnection,
	'invitationPong': notifyInvitationPong,
	'invitationTournamentPong': notifyInvitationTournamentPong,
	'invitationTowerDefense': notifyInvitationTowerDefense,
	'loseTournament': loseTournament,
	'winTournament': winTournament,
	'winBracket': winBracket,
}