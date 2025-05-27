import { z } from 'zod'

export const verifySchema = z.object({
	secret: z.string().regex(/^\d{6}$/, {
		message: 'only 6 digits are allowed',
	}),
	nickName: z.string().min(3, '3 character or more for the nickname')
		.regex(/^[a-zA-Z0-9]+$/, 'only alphanumeric character accepted for the nickname')
})

export const pictureSchema = z.object({
	pictureURL: z.string().regex(
		/^data:image\/(jpeg|png);base64/,
		'only jpeg and png are allowed'
	)
})

export const idArraySchema = z.object({
	ids: z.array(z.number())
})


export const signUpSchema = z.object({
	nickName: z.string().min(3, '3 character or more for the nickname')
		.max(15, '15 character or less for the nickname')
		.regex(/^[a-zA-Z0-9]+$/, 'only alphanumeric character accepted for the nickname'),
	email: z.string().email('Invalid email format'),
	password: z.string().min(6, '6 character or more for the password'),
})

export const signInSchema = z.object({
	nickName: z.string().min(3, '3 character or more for the nickname')
		.max(15, '15 character or less for the nickname')
		.regex(/^[a-zA-Z0-9]+$/, 'only alphanumeric character accepted for the nickname'),
	password: z.string().min(6, '6 character or more for the password'),
})

export const passwordSchema = z.object({
	password: z.string().min(6, '6 character or more for the password')
})

export const nickNameSchema = z.object({
	nickName: z.string().min(3, '3 character or more for the nickname')
		.max(15, '15 character or less for the nickname')
		.regex(/^[a-zA-Z0-9]+$/, 'only alphanumeric character accepted for the nickname'),
})

export const invitationGameSchema = z.object({
	id: z.number(),
	roomId: z.number(),
	game: z.enum([`pong`, `towerDefense`])
})

export const notifySchema = z.object({
	ids: z.array(z.number()),
	event: z.string(),
	data: z.any()
})

export const mmrSchema = z.object({
	mmr: z.number()
})

export * from './schema.js'