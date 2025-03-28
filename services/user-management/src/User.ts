import Database from "better-sqlite3";
import bcrypt from "bcrypt";
import jwt from 'jsonwebtoken';
import {connectedUsers} from "./api.js";

export const Client_db = new Database('client.db')  // Importation correcte de sqlite

Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nickName TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL,
        google_id INTEGER
    )
`);

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
    id: string;

    constructor(id: string) {
        this.id = id;

        if (!Client_db.prepare("SELECT * FROM Client WHERE id = ?").get(this.id)) {
            throw new Error(`${this.id} not found`);
        }
    }

    static async addClient(nickName: string, email: string, password: string): Promise<{success: boolean, result: string}> {
        if (Client_db.prepare("SELECT * FROM Client WHERE nickName = ?").get(nickName)) {
            return {success: false, result: 'nickName'};
        }
        if (Client_db.prepare("SELECT * FROM Client WHERE email = ?").get(email)) {
            return {success: false, result: 'email'};
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const res = Client_db.prepare("INSERT INTO Client (nickName, email, password) VALUES (?, ?, ?)")
            .run(nickName, email, hashedPassword);
        if (res.changes === 0) {
            throw new Error(`User not inserted`);
        }
        const id : number | bigint = res.lastInsertRowid
        const stringId: string = id.toString();

        console.log("new user added");
        return {success: true, result: this.makeToken(stringId)};
    }

    static async login(nickName: string, password: string): Promise<{code: number, result: string}> {
        const userData = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id: string};
        if (!userData)
            return {code: 409, result: "this nickName doesn't exist"};

        const client = new User(userData.id)

        if (!await client.isPasswordValid(password))
            return {code: 401, result: "invalid password"};
        return {code: 201, result: User.makeToken(client.id)}
    }

    async isPasswordValid(password: string): Promise<boolean> {
        const userData = Client_db.prepare("SELECT password FROM Client WHERE id = ?").get(this.id) as { password: string } | undefined;
        if (!userData) {
            throw new Error(`Database Error: cannot find password from ${this.id}`);
        }
        return (await bcrypt.compare(password, userData.password))
    }

    private static makeToken(id: string): string {
        const token = jwt.sign({id: id}, 'secret_key', {expiresIn: '1h'})
        return (token);
    }

    getProfile(): { nickName: string, email: string } {
        const userData = Client_db.prepare("SELECT nickName, email FROM Client WHERE id = ?").get(this.id) as { nickName: string, email: string } | undefined;
        if (!userData) {
            throw new Error(`Database Error: cannot find data from ${this.id}`);
        }
        return {nickName: userData.nickName, email: userData.email};
    }

    sendNotification() {
        const res = connectedUsers.get(this.id);
        if (!res)
            return console.log("Server error: res not found in connectedUsers");
        res.sse({data: JSON.stringify({event: "invite", data: "teeest"})});
    }

    addFriend(nickName: string): {code: number, message: string} {
        const friendId = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id :number};
        if (!friendId)
            return {code: 409, message: "this nickName doesn't exist"};

        const checkStatus = Client_db.prepare("SELECT status FROM FriendList WHERE userA_id = ? AND userB_id = ?").get(friendId.id, this.id) as {status: string};
        if (checkStatus?.status == 'accepted')
            return {code: 409, message: "Friend already added"};

        else if (checkStatus?.status == 'pending') {
            Client_db.prepare(`UPDATE FriendList SET status = 'accepted' WHERE userA_id = ? AND userB_id = ?`).run(friendId.id, this.id);
            return {code: 201, message: "Friend invitation accepted"};
        }

        const res = Client_db.prepare(`INSERT OR IGNORE INTO FriendList (userA_id, userB_id, status) VALUES (?, ?, ?)`).run(this.id, friendId.id, 'pending');
        if (res.changes === 0)
            return {code: 409, message: "Friend invitation already sent"};

        return {code: 201, message: `friend invitation sent successfully`};
    }

    removeFriend(nickName: string): {code: number, message: string} {
        const friendId = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id :number};
        if (!friendId)
            return {code: 409, message: "this nickName doesn't exist"};

        const checkStatus = Client_db.prepare("DELETE FROM FriendList WHERE (userA_id = ? AND userB_id = ?) OR (userB_id = ? AND userA_id = ?)").run(friendId.id, this.id, friendId.id, this.id);
        if (!checkStatus.changes)
            return {code: 409, message: "This user isn't your friendList"};
        return {code: 201, message: "Friend removed"};
    }

    getFriendList(): {acceptedIds: number[], pendingIds: number[], receivedIds: number[]} {
        const accepted = Client_db.prepare(`SELECT userA_id FROM FriendList WHERE userB_id = ? AND status = 'accepted' UNION SELECT userB_id FROM FriendList WHERE userA_id = ? AND status = 'accepted'`).all(this.id, this.id) as {userA_id?: number, userB_id?: number }[];
        const pending = Client_db.prepare(`SELECT userB_id FROM FriendList WHERE userA_id = ? AND status = 'pending'`).all(this.id) as {userB_id: number}[];
        const invited = Client_db.prepare(`SELECT userA_id FROM FriendList WHERE userB_id = ? AND status = 'pending'`).all(this.id) as {userA_id: number}[];

        const acceptedIds = accepted.map(row => row.userA_id ?? row.userB_id).filter(id => id !== undefined);
        const pendingIds = pending.map(row => row.userB_id).filter(id => id !== undefined);
        const receivedIds = invited.map(row => row.userA_id).filter(id => id !== undefined);

        return {acceptedIds, pendingIds, receivedIds};
    }

}