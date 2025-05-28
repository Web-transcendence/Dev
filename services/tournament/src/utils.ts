import { fetch } from 'undici'
import { UnauthorizedError } from './error.js'
import { INTERNAL_PASSWORD } from './api.js'


export const authUser = async (id: number) => {
	const result = await fetch(`http://user-management:5000/authId/${id}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `${INTERNAL_PASSWORD}`
		},
	})
	if (!result.ok)
		throw new UnauthorizedError(`this id doesn't exist in database`, `this client doesn't exist`)
}

export const fetchNotifyUser = async (ids: number[], event: string, data: any) => {
	await fetch('http://user-management:5000/notify', {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			'Authorization': `${INTERNAL_PASSWORD}`
		},
		body: JSON.stringify({ ids: ids, event: event, data: data }),
	})
}