import Database from "better-sqlite3";
import bcrypt from "bcrypt";
import jwt from 'jsonwebtoken';
import {connectedUsers} from "./api.js";

export const Client_db = new Database('client.db')  // Importation correcte de sqlite

const BASE64_UNDEFINED_PICTURES = 'A DEFINIR' // attention !!

Client_db.exec(`
    CREATE TABLE IF NOT EXISTS Client (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nickName TEXT NOT NULL,
        email UNIQUE NOT NULL COLLATE NOCASE,
        password TEXT NOT NULL,
        google_id INTEGER,
        pictureProfile TEXT DEFAULT NULL,
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


    /**
     * check in the db if this nickname exist. if yes it call .isPasswordValid() to
     *      check if the password match with the password given.
     *
     * @param nickName
     * @param password
     * @return JWT on success
     */
    static async login(nickName: string, password: string): Promise<{code: number, result: string}> {
        const userData = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id: string};
        if (!userData)
            return {code: 409, result: "this nickName doesn't exist"};

        const client = new User(userData.id)

        if (!await client.isPasswordValid(password))
            return {code: 401, result: "invalid password"};
        return {code: 201, result: User.makeToken(client.id)}
    }

    /** recover the hashed password from the db and use bcrypt tu compare with the one given.
     *
     * @param password
     */
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

    updatePictureProfile(pictureURL: string): {code: number, message: string} {
        const change = Client_db.prepare(`UPDATE Client SET pictureProfile = ? WHERE id = ?`).run(pictureURL, this.id);
        if (!change || !change.changes)
            throw new Error(`Database Error: cannot update pictureProfile changes for ${this.id}`);
        return {code: 200, message: "picture updated"};
    }

    getpictureProfile(): {code: number, message: string} {
        const userData = Client_db.prepare(`SELECT pictureProfile FROM Client WHERE id = ?`).get(this.id) as {base64Picture: string};
        if (!userData)
            throw new Error(`Database Failed`);
        if (!userData.base64Picture)
            return {code: 200, message: BASE64_UNDEFINED_PICTURES};
        return {code: 200, message: userData.base64Picture};
    }

    /**
     * recover the id of the friend, check his friendship status in db, if it doesn't exist it add with status = pending,
     *      if it is pending and the user is userB_id, it update status to accepted, it it is userA_id nothing happens.
     *
     * @param nickName
     * @return a status for the client
     */
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

    /**
     * recover the id of the client, remove it. if there wasn't friend nothing happens (checkstatus.changes set to 0)
     *
     * @param nickName
     */
    removeFriend(nickName: string): {code: number, message: string} {
        const friendId = Client_db.prepare("SELECT id FROM Client WHERE nickName = ?").get(nickName) as {id :number};
        if (!friendId)
            return {code: 409, message: "this nickName doesn't exist"};

        const checkStatus = Client_db.prepare("DELETE FROM FriendList WHERE (userA_id = ? AND userB_id = ?) OR (userB_id = ? AND userA_id = ?)").run(friendId.id, this.id, friendId.id, this.id);
        if (!checkStatus.changes)
            return {code: 409, message: "This user isn't your friendList"};
        return {code: 201, message: "Friend removed"};
    }

    /**
     * recover friend by status:
     *      - accepted when each of them accepted the friendship
     *      - pending when the user is waiting the friend to accept
     *      - received when the user received an invitation by another user and he didn't accept yet
     *
     * @return an object with three array of id. One for each type of friend.
     */
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