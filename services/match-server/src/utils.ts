import { INTERNAL_PASSWORD } from './api.js'

export const fetchIdByNickName = async (nickName: string): Promise<number> => {
	if (nickName === 'IA')
		return (-2)
	const response = await fetch(`http://user-management:5000/idByNickName/${nickName}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'authorization': `${INTERNAL_PASSWORD}`
		},
	})
	if (!response.ok) {
		throw new Error(`this nickname doesn't exist`)
	}
	const { id } = await response.json() as { id: number }
	return id
}

export const fetchMmrById = async (dbId: number): Promise<number> => {
	if (dbId === -1)
		return (1200)
	const response = await fetch(`http://user-management:5000/pong/mmrById/${dbId}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'authorization': `${INTERNAL_PASSWORD}`
		},
	})
	if (!response.ok) {
		throw new Error(`this nickname doesn't exist`)
	}
	const { mmr } = await response.json() as { mmr: number }
	return mmr
}

export const updateMmrById = async (dbId: number, currentMmr: number, win: boolean): Promise<void> => {
	if (dbId === -1) // Si Guest, on ne fait rien
		return
	let newMmr = currentMmr + 10 * (win ? 1 : -1)
	if (newMmr < 0)
		newMmr = 0
	const response = await fetch(`http://user-management:5000/mmrById/${dbId}`, {
		method: 'PUT', // ou 'PATCH' selon ton API
		headers: {
			'Content-Type': 'application/json',
			'authorization': `${INTERNAL_PASSWORD}`
		},
		body: JSON.stringify({ mmr: newMmr })
	})
	if (!response.ok) {
		throw new Error(`Failed to update MMR for player with id ${dbId}`)
	}
} // Attention c'est ChatGpt qui a fait ça, je ne sais pas si ça marche
// En gros j'appelle cette fonction avec le id du joueur et son mmr actuel, et je lui dis si il a gagné ou perdu
// Si lose il perd 10 de mmr, si win il en gagne 10
// J'envoie le nouveau mmr a userManagement

export const fetchNotifyUser = async (ids: number[], event: string, data: any) => {
	const response = await fetch('http://user-management:5000/notify', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `${INTERNAL_PASSWORD}`
		},
		body: JSON.stringify({ ids: ids, event: event, data: data }),
	})
}

export const fetchPlayerWin = async (winnerId: number) => {
	await fetch(`http://tournament:7000/userWin/${winnerId}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `${INTERNAL_PASSWORD}`
		}
	})
}