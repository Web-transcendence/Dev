import Database from "better-sqlite3"
import bcrypt from "bcrypt"
import jwt from 'jsonwebtoken'
import {connectedUsers} from "./api.js"
import speakeasy, {GeneratedSecret} from "speakeasy"
import QRCode from "qrcode"
import {ConflictError, DataBaseError, ServerError, UnauthorizedError} from "./error.js";
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
            throw new ServerError(`Client not found`, 404)
        }
    }


    /**
     * check if nickName and email aren't in the db, hash the password
     *      and add the data in the db. Then the db return the id of
     *      this client
     *
     * @param nickName
     * @param email
     * @param password
     * @return boolean about status and if it success the JWT
     */
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


    /**
     * check in the db if this nickname exist. if yes it call .isPasswordValid() to
     *      check if the password match with the password given.
     *
     * @param nickName
     * @param password
     * @return JWT on success
     */
    static async login(nickName: string, password: string): Promise<string> {
        const id: number = this.getIdbyNickName(nickName)

        const client = new User(id)

        if (!await client.isPasswordValid(password))
            throw new UnauthorizedError(`bad password`, 'wrong password')

        const data = Client_db.prepare("SELECT activated2fa FROM Client WHERE id = ?").get(id) as { activated2fa: boolean } | undefined
        if (!data)
            throw new DataBaseError(`User with ID ${id} not found`, `internal error system`, 500)
        if (data.activated2fa)
            return ''
        return User.makeToken(client.id)
    }

    /** recover the hashed password from the db and use bcrypt tu compare with the one given.
     *
     * @param password
     */
    async isPasswordValid(password: string): Promise<boolean> {
        const userData = Client_db.prepare("SELECT password FROM Client WHERE id = ?").get(this.id) as { password: string } | undefined
        if (!userData)
            throw new DataBaseError(`User with ID ${this.id} not found`, 'internal error system', 500)

        return await bcrypt.compare(password, userData.password)
    }

    private static makeToken(id: number): string {
        const token = jwt.sign({id: id}, 'secret_key', {expiresIn: '1h'})
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
            throw new ConflictError("2fa isn't activated", 'internal error system')

        const verified = speakeasy.totp.verify({
            secret: secret.secret_key,
            encoding: 'base32',
            token: token,
            window: 1
        })
        if (!verified)
            throw new UnauthorizedError(`invalid secret key for 2fa`, 'bad 2fa code')
        Client_db.prepare(`UPDATE Client SET activated2fa = ? WHERE id = ?`).run(1, this.id)
        return User.makeToken(this.id)
    }

    getProfile(): { nickName: string, email: string } {
        const userData = Client_db.prepare("SELECT nickName, email FROM Client WHERE id = ?").get(this.id) as { nickName: string, email: string } | undefined
        if (!userData) {
            throw new DataBaseError(`User with ID ${this.id} not found`, `internal error system`, 500)
        }
        return {nickName: userData.nickName, email: userData.email}
    }

    updatePictureProfile(pictureURL: string) {
        const change = Client_db.prepare(`UPDATE Client SET pictureProfile = ? WHERE id = ?`).run(pictureURL, this.id);
        if (!change || !change.changes)
            throw new DataBaseError(`cannot upload the picture in the db`, 'error 500: internal error system', 500)
    }

    getPictureProfile(): string {
        const userData = Client_db.prepare(`SELECT pictureProfile FROM Client WHERE id = ?`).get(this.id) as {pictureProfile: string} | undefined;
        if (!userData)
            throw new DataBaseError(`should not happen`, 'internal error system', 500)
        if (!userData.pictureProfile)
            return '';
        return userData.pictureProfile;
    }

    /**
     * recover the id of the friend, check his friendship status in db, if it doesn't exist it add with status = pending,
     *      if it is pending and the user is userB_id, it update status to accepted, it it is userA_id nothing happens.
     *
     * @param nickName
     * @return a status for the client
     */


    static getIdbyNickName(nickName: string): number {
        const userData = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id: number} | undefined
        if (!userData)
            throw new DataBaseError(`id not found for nickName: ${nickName}`, `This nickname doesn't exist`, 404)
        return userData.id
    }

    getNickNameById(id: number): string {
        const userData = Client_db.prepare("SELECT nickName FROM Client WHERE id = ?").get(id) as {nickName: string} | undefined
        if (!userData)
            throw new DataBaseError(`nickName not found for id ${id}`, 'internal error system', 500)
        return (userData.nickName)
    }



    async sseHandler(req: FastifyRequest, res: FastifyReply) {

        if (connectedUsers.has(this.id))
            return res.status(100).send()
        connectedUsers.set(this.id, res)
        await connection(this.id)
        console.log('sse connected : id', this.id)
        const message: EventMessage = {event: "ping"}
        res.sse({data: JSON.stringify(message)})
        const interval = setInterval(() => {
            const message: EventMessage = {event: "ping"}
            res.sse({data: JSON.stringify(message)})
        }, 15000)

        req.raw.on('close', async() => {
            clearInterval(interval)
            if (connectedUsers.has(this.id)) {
                console.log('sse disconnected client = ' + this.id)
                await disconnect(this.id)
                connectedUsers.delete(this.id)
            }
        })
    }



}
