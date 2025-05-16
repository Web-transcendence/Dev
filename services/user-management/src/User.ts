import Database from "better-sqlite3"
import bcrypt from "bcrypt"
import {connectedUsers, app} from "./api.js"
import speakeasy, {GeneratedSecret} from "speakeasy"
import QRCode from "qrcode"
import {ConflictError, DataBaseError, NotFoundError, ServerError, UnauthorizedError} from "./error.js";
import {EventMessage} from "fastify-sse-v2";
import { FastifyReply, FastifyRequest } from "fastify"
import {connection, disconnect} from "./serverSentEvent.js";


export const Client_db = new Database('client.db')  // Importation correcte de sqlite



Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nickName TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL,
        google_id INTEGER,
        secret_key TEXT DEFAULT NULL,
        pictureProfile TEXT DEFAULT NULL,
        activated2fa BOOLEAN DEFAULT NULL
    )
`)


export class User {
    id: number

    constructor(id: number) {
        this.id = id
        if (!Client_db.prepare("SELECT * FROM Client WHERE id = ?").get(this.id)) {
            throw new NotFoundError(`Client not found`, `Client not found`)
        }
    }

    // AUTHENTIFICATION AND CONNECTION //


    static async addClient(nickName: string, email: string, password: string): Promise<string> {
        if (Client_db.prepare("SELECT * FROM Client WHERE nickName = ?").get(nickName))
            throw new ConflictError(`${nickName} is already taken`, `This nickname is already taken`)
        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email))
            throw new ConflictError(`${email} is already taken`, `This email is already taken`)

        const hashedPassword: string = await bcrypt.hash(password, 10)

        const res = Client_db.prepare("INSERT INTO Client (nickName, email, password) VALUES (?, ?, ?)")
            .run(nickName, email, hashedPassword)
        if (res.changes === 0)
            throw new DataBaseError(`client couldn't be inserted`, 'error 500 : internal error system', 500)

        const id : number = Number(res.lastInsertRowid)

        return this.makeToken(id)
    }

    static async login(nickName: string, password: string): Promise<string | null> {
        const id: number = this.getIdbyNickName(nickName)

        const client = new User(id)

        if (!await client.isPasswordValid(password))
            throw new UnauthorizedError(`bad password`, 'wrong password')

        if (connectedUsers.has(id))
            throw new ConflictError("user try to connect on different sessions", "you are already connected on an other session")

        const data = Client_db.prepare("SELECT activated2fa FROM Client WHERE id = ?").get(id) as { activated2fa: boolean } | undefined
        if (!data)
            throw new DataBaseError(`User with ID ${id} not found`, `internal error system`, 500)
        if (data.activated2fa)
            return null
        return User.makeToken(client.id)
    }

    async isPasswordValid(password: string): Promise<boolean> {
        const userData = Client_db.prepare("SELECT password FROM Client WHERE id = ?").get(this.id) as { password: string } | undefined
        if (!userData)
            throw new DataBaseError(`User with ID ${this.id} not found`, 'internal error system', 500)

        return await bcrypt.compare(password, userData.password)
    }

    static makeToken(id: number): string {
        const token = app.jwt.sign({id: id})
        return (token)
    }

    async generateSecretKey(): Promise<string> {
        const data = Client_db.prepare("SELECT secret_key FROM Client WHERE id = ?").get(this.id) as { secret_key: string } | undefined
        if (!data)
            throw new DataBaseError(`User with ID ${this.id} not found`, 'internal error system', 500)
        if (data.secret_key) {
            const otpauthUrl = speakeasy.otpauthURL({
                secret: data.secret_key,
                label: 'transcendence',
                issuer: 'master',
                encoding: 'base32'
            });
            return await QRCode.toDataURL(otpauthUrl);
        }

        const secret:GeneratedSecret = speakeasy.generateSecret()

        if (!secret || !secret.otpauth_url)
            throw new ServerError(`speakeasy failed to create a secret key`, 500)

        Client_db.prepare("UPDATE Client set secret_key = ? where id = ?").run(secret.base32, this.id)

        return await QRCode.toDataURL(secret.otpauth_url)
    }

    verify(token: string): string {
        const secret = Client_db.prepare("SELECT secret_key FROM Client WHERE id = ?").get(this.id) as { secret_key: string } | undefined
        if (!secret)
            throw new DataBaseError(`secret key not found for id: ${this.id}`, 'internal error system', 500)
        if (!secret.secret_key)
            throw new ConflictError("2fa isn't activated", `you have to activate the 2fa before verifying it`)

        const verified = speakeasy.totp.verify({
            secret: secret.secret_key,
            encoding: 'base32',
            token: token,
            window: 1
        })
        if (!verified)
            throw new UnauthorizedError(`invalid secret key for 2fa`, 'wrong code')
        Client_db.prepare(`UPDATE Client SET activated2fa = ? WHERE id = ?`).run(1, this.id)
        return User.makeToken(this.id)
    }

    // SETTER AND GETTER //

    async setPassword(newPassword: string) {
        const hashedPassword: string = await bcrypt.hash(newPassword, 10)

        if (!Client_db.prepare("UPDATE Client set password = ? where id = ?").run(hashedPassword, this.id))
            throw new DataBaseError('cannot insert new password', 'internal error system', 500)
    }

    async setNickname(newNickName: string) {
        if (Client_db.prepare("SELECT * FROM Client WHERE nickName = ?").get(newNickName))
            throw new ConflictError("An user try to set his nickName to an already used nickname", "Nickname already used")

        if (!Client_db.prepare("UPDATE Client set nickName = ? where id = ?").run(newNickName, this.id))
            throw new DataBaseError('cannot insert new password', 'internal error system', 500)
    }

    updatePictureProfile(pictureURL: string) {
        const change = Client_db.prepare(`UPDATE Client SET pictureProfile = ? WHERE id = ?`).run(pictureURL, this.id);
        if (!change || !change.changes)
            throw new DataBaseError(`cannot upload the picture in the db`, 'error 500: internal error system', 500)
    }


    getProfile(): { id:number,  nickName: string, email: string, avatar: string } {
        const userData = Client_db.prepare("SELECT nickName, email, pictureProfile FROM Client WHERE id = ?").get(this.id) as { nickName: string, email: string, pictureProfile: string } | undefined
        if (!userData) {
            throw new DataBaseError(`User with ID ${this.id} not found`, `internal error system`, 500)
        }
        return {id: this.id, nickName: userData.nickName, email: userData.email, avatar: userData.pictureProfile}
    }

    publicData(): {id:number,  nickName: string, avatar: string, online: boolean} {
        const data = Client_db.prepare("SELECT nickName, pictureProfile FROM Client WHERE id = ?").get(this.id) as { nickName: string, pictureProfile: string } | undefined
        if (!data)
            throw new DataBaseError(`User with ID ${this.id} not found`, 'internal error system', 500)

        return {
            id: this.id,
            nickName: data.nickName,
            avatar: data.pictureProfile,
            online: connectedUsers.has(this.id)
        }
    }


    getPictureProfile(): string {
        const userData = Client_db.prepare(`SELECT pictureProfile FROM Client WHERE id = ?`).get(this.id) as {pictureProfile: string} | undefined;
        if (!userData)
            throw new DataBaseError(`should not happen`, 'internal error system', 500)
        if (!userData.pictureProfile)
            return '';
        return userData.pictureProfile;
    }


    static getIdbyNickName(nickName: string): number {
        const userData = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id: number} | undefined
        if (!userData)
            throw new DataBaseError(`id not found for nickName: ${nickName}`, `This nickname doesn't exist`, 404)
        return userData.id
    }

    // SSE //

    async sseHandler(req: FastifyRequest, res: FastifyReply) {
        if (connectedUsers.has(this.id))
            return res.status(409).send({error: "already connected by sse"})
        connectedUsers.set(this.id, res)
        await connection(this.id)
        console.log(`open`)
        const message: EventMessage = {event: "ping"}
        res.sse({data: JSON.stringify(message)})
        const interval = setInterval(() => {
            const message: EventMessage = {event: "ping"}
            res.sse({data: JSON.stringify(message)})
        }, 15000)

        req.raw.on('close', async() => {
            console.log(`close`)
            clearInterval(interval)
            if (connectedUsers.has(this.id)) {
                console.log('sse disconnected client = ' + this.id)
                await disconnect(this.id)
                connectedUsers.delete(this.id)
            }
        })
    }



}
