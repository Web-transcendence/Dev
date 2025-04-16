import Database from "better-sqlite3"
import bcrypt from "bcrypt"
import jwt from 'jsonwebtoken'
import {connectedUsers, tournamentSessions} from "./api.js"
import speakeasy, {GeneratedSecret} from "speakeasy"
import QRCode from "qrcode"
import {tournament} from "./tournament.js"
import {number} from "zod"
import {ConflictError, DataBaseError, ServerError, UnauthorizedError} from "./error.js";
import {type} from "node:os";

export const Client_db = new Database('client.db')  // Importation correcte de sqlite

interface UserData {
    id?: number
}

Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nickName TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL,
        google_id INTEGER,
        secret_key TEXT DEFAULT NULL
    )
`)

Client_db.exec(`
    CREATE TABLE IF NOT EXISTS FriendList (
        userA_id INTEGER NOT NULL,
        userB_id INTEGER NOT NULL,
        status TEXT check(status IN ('pending', 'accepted')) DEFAULT ('pending'),
        PRIMARY KEY (userA_id, userB_id),
        FOREIGN KEY (userA_id) REFERENCES Client(id) ON DELETE CASCADE,
        FOREIGN KEY (userB_id) REFERENCES Client(id) ON DELETE CASCADE
    )
`)

export class User {
    id: number

    constructor(id: number) {
        this.id = id
        console.log(id,typeof(id))
        if (!Client_db.prepare("SELECT * FROM Client WHERE id = ?").get(this.id)) {
            throw new ServerError(`Client not found`, 404)
        }
    }

    static getIdbyNickName(nickName: string): number {
        const userData = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id: number} | undefined
        if (!userData)
            throw new DataBaseError(`id not found for nickName: ${nickName}`, 404)
        return userData.id
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
            throw new ConflictError(`${nickName} is already taken`)
        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email))
            throw new ConflictError(`${nickName} is already taken`)

        const hashedPassword: string = await bcrypt.hash(password, 10)

        const res = Client_db.prepare("INSERT INTO Client (nickName, email, password) VALUES (?, ?, ?)")
            .run(nickName, email, hashedPassword)
        if (res.changes === 0)
            throw new DataBaseError(`client couldn't be inserted`, 500)

        const id : number = number(res.lastInsertRowid)

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
            throw new UnauthorizedError(`bad password`)
        return User.makeToken(client.id)
    }

    /** recover the hashed password from the db and use bcrypt tu compare with the one given.
     *
     * @param password
     */
    async isPasswordValid(password: string): Promise<boolean> {
        const userData = Client_db.prepare("SELECT password FROM Client WHERE id = ?").get(this.id) as { password: string } | undefined
        if (!userData)
            throw new DataBaseError(`User with ID ${this.id} not found, which should not happen`, 500)

        return await bcrypt.compare(password, userData.password)
    }

    private static makeToken(id: number): string {
        const token = jwt.sign({id: id}, 'secret_key', {expiresIn: '1h'})
        return (token)
    }

    async generateSecretKey(): Promise<string> {
        const data = Client_db.prepare("SELECT secret_key FROM Client WHERE id = ?").get(this.id) as { secret_key: string } | undefined
        if (!data)
            throw new DataBaseError(`User with ID ${this.id} not found, which should not happen`, 500)
        if (data.secret_key)
            throw new ConflictError(`secret key is already made`)

        const secret:GeneratedSecret = speakeasy.generateSecret()

        if (!secret || !secret.otpauth_url)
            throw new ServerError(`speakeasy failed to create a secret key`, 500)

        Client_db.prepare("UPDATE Client set secret_key = ? where id = ?").run(secret.base32, this.id)

        return await QRCode.toDataURL(secret.otpauth_url)
    }

    verify(token: string) {
        const secret = Client_db.prepare("SELECT secret_key FROM Client WHERE id = ?").get(this.id) as { secret_key: string } | undefined
        if (!secret)
            throw new DataBaseError(`secret key not found for id: ${this.id}`, 500)
        if (!secret.secret_key)
            throw new ConflictError("2fa isn't activated")

        const verified = speakeasy.totp.verify({
            secret: secret.secret_key,
            encoding: 'base32',
            token: token,
            window: 1
        })
        if (!verified)
            throw new UnauthorizedError(`invalid secret key for 2fa`)
    }

    getProfile(): { nickName: string, email: string } {
        const userData = Client_db.prepare("SELECT nickName, email FROM Client WHERE id = ?").get(this.id) as { nickName: string, email: string } | undefined
        if (!userData) {
            throw new DataBaseError(`User with ID ${this.id} not found, which should not happen`, 500)
        }
        return {nickName: userData.nickName, email: userData.email}
    }

    sendNotification() {
        const res = connectedUsers.get(this.id)
        if (!res)
            return console.log("Server error: res not found in connectedUsers")
        res.sse({data: JSON.stringify({event: "invite", data: "teeest"})})
    }


    /**
     * recover the id of the friend, check his friendship status in db, if it doesn't exist it add with status = pending,
     *      if it is pending and the user is userB_id, it update status to accepted, it it is userA_id nothing happens.
     *
     * @param nickName
     * @return a status for the client
     */
    addFriend(nickName: string): string {
        const friendId = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id :number} | undefined
        if (!friendId)
            throw new DataBaseError(`id not found for this friend nickName: ${nickName}`, 404)

        const checkStatus = Client_db.prepare("SELECT status FROM FriendList WHERE userA_id = ? AND userB_id = ?").get(friendId.id, this.id) as {status: string}
        if (checkStatus?.status == 'accepted')
            throw new ConflictError(`This friend is already in friendList`)

        else if (checkStatus?.status == 'pending') {
            Client_db.prepare(`UPDATE FriendList SET status = 'accepted' WHERE userA_id = ? AND userB_id = ?`).run(friendId.id, this.id)
            return `Friend invitation accepted`
        }

        const res = Client_db.prepare(`INSERT OR IGNORE INTO FriendList (userA_id, userB_id, status) VALUES (?, ?, ?)`).run(this.id, friendId.id, 'pending')
        if (res.changes === 0)
            throw new ConflictError(`Friend invitation already sent`)

        return `Friend invitation sent successfully`
    }

    /**
     * recover the id of the client, remove it. if there wasn't friend nothing happens (checkstatus.changes set to 0)
     *
     * @param nickName
     */
    removeFriend(nickName: string) {
        const friendId = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id :number} | undefined
        if (!friendId)
            throw new DataBaseError(`id not found for this friend nickName: ${nickName}`, 404)

        const checkStatus = Client_db.prepare("DELETE FROM FriendList WHERE (userA_id = ? AND userB_id = ?) OR (userB_id = ? AND userA_id = ?)").run(friendId.id, this.id, friendId.id, this.id)
        if (!checkStatus.changes)
            throw new ConflictError(`This user isn't in your friendList`)
    }

    /**
     * recover friend by status:
     *      - accepted when each of them accepted the friendship
     *      - pending when the user is waiting the friend to accept
     *      - received when the user received an invitation by another user and he didn't accept yet
     *
     * @return an object with three array of id. One for each type of friend.
     */
    getFriendList(): {acceptedNickName: string[], pendingNickName: string[], receivedNickName: string[]} {
        const accepted = Client_db.prepare(`SELECT userA_id FROM FriendList WHERE userB_id = ? AND status = 'accepted' UNION SELECT userB_id FROM FriendList WHERE userA_id = ? AND status = 'accepted'`).all(this.id, this.id) as {userA_id?: number, userB_id?: number }[]
        const pending = Client_db.prepare(`SELECT userB_id FROM FriendList WHERE userA_id = ? AND status = 'pending'`).all(this.id) as {userB_id: number}[]
        const invited = Client_db.prepare(`SELECT userA_id FROM FriendList WHERE userB_id = ? AND status = 'pending'`).all(this.id) as {userA_id: number}[]

        const acceptedIds = accepted.map(row => row.userA_id ?? row.userB_id).filter(id => id !== undefined)
        const pendingIds = pending.map(row => row.userB_id).filter(id => id !== undefined)
        const receivedIds = invited.map(row => row.userA_id).filter(id => id !== undefined)

        const acceptedNickName = acceptedIds.map(row => this.getNickNameById(row))
        const pendingNickName = pendingIds.map(row => this.getNickNameById(row))
        const receivedNickName = receivedIds.map(row => this.getNickNameById(row))


        return {acceptedNickName, pendingNickName, receivedNickName}
    }

    getNickNameById(id: number): string {
        const userData = Client_db.prepare("SELECT nickName FROM Client WHERE id = ?").get(id) as {nickName: string} | undefined
        if (!userData)
            throw new DataBaseError(`nickName not found for id ${id}`, 500)
        return (userData.nickName)
    }

    createTournament(): tournament{
        for (const [id, tournament] of tournamentSessions)
            if (tournament.hasParticipant(this.id))
                throw new ConflictError(`this user is already in a tournament`)
        return new tournament(this.id)
    }

    getActualTournament(): tournament | null {
        for (const [id, tournament] of tournamentSessions)
            if (tournament.hasParticipant(this.id))
                return tournament
        return null
    }
}
