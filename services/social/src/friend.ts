import Database from "better-sqlite3"
import {ConflictError} from "./error.js";
import {fetchId} from "./utils.js"
import {fetchNotifyUser} from "./utils.js";

export const Friend_db = new Database('friend.db')  // Importation correcte de sqlite


type FriendList = {
    acceptedIds: number[];
    pendingIds: number[];
    receivedIds: number[];
};


Friend_db.exec(`
    CREATE TABLE IF NOT EXISTS FriendList (
        userA_id INTEGER NOT NULL,
        userB_id INTEGER NOT NULL,
        status TEXT check(status IN ('pending', 'accepted')) DEFAULT ('pending'),
        PRIMARY KEY (userA_id, userB_id)
    )
`)


export function getFriendList(id: number): FriendList {
    const accepted = Friend_db.prepare(`SELECT userA_id FROM FriendList WHERE userB_id = ? AND status = 'accepted' UNION SELECT userB_id FROM FriendList WHERE userA_id = ? AND status = 'accepted'`).all(id, id) as {userA_id?: number, userB_id?: number }[]
    const pending = Friend_db.prepare(`SELECT userB_id FROM FriendList WHERE userA_id = ? AND status = 'pending'`).all(id) as {userB_id: number}[]
    const invited = Friend_db.prepare(`SELECT userA_id FROM FriendList WHERE userB_id = ? AND status = 'pending'`).all(id) as {userA_id: number}[]

    const acceptedIds = accepted.map(row => row.userA_id ?? row.userB_id).filter(id => id !== undefined) as number[]
    const pendingIds = pending.map(row => row.userB_id).filter(id => id !== undefined)
    const receivedIds = invited.map(row => row.userA_id).filter(id => id !== undefined)

    return {acceptedIds, pendingIds, receivedIds}
}

export async function addFriend(id: number, nickName: string) {
    const friendId: number = await fetchId(nickName)

    if (friendId === id)
        throw new ConflictError(`user tryed to add itself`, `cannot add yourself`)

    const checkStatus = Friend_db.prepare("SELECT status FROM FriendList WHERE userA_id = ? AND userB_id = ?").get(friendId, id) as {status: string}
    if (checkStatus?.status == 'accepted')
        throw new ConflictError(`This friend is already in friendList`, `This nickname is already in your friendlist`)

    else if (checkStatus?.status == 'pending') {
        Friend_db.prepare(`UPDATE FriendList SET status = 'accepted' WHERE userA_id = ? AND userB_id = ?`).run(friendId, id)
        await fetchNotifyUser([friendId], 'newFriend', {id: id})
        return `Friend invitation accepted`
    }

    const res = Friend_db.prepare(`INSERT OR IGNORE INTO FriendList (userA_id, userB_id, status) VALUES (?, ?, ?)`).run(id, friendId, 'pending')
    if (res.changes === 0)
        throw new ConflictError(`Friend invitation already sent`, `You already send an invitation to this nickname`)
    await fetchNotifyUser([friendId], 'friendInvitation', {id: id})
    return `Friend invitation sent successfully`
}

/**
 * recover the id of the client, remove it. if there wasn't friend nothing happens (checkstatus.changes set to 0)
 *
 * @param nickName
 */
export async function removeFriend(id: number, nickName: string) {
    const friendId: number = await fetchId(nickName)

    if (friendId === id)
        throw new ConflictError('cannot remove itself', `internal server error`)

    const checkStatus = Friend_db.prepare("DELETE FROM FriendList WHERE (userA_id = ? AND userB_id = ?) OR (userB_id = ? AND userA_id = ?)").run(friendId, id, friendId, id)
    if (!checkStatus.changes)
        throw new ConflictError(`This user isn't in your friendList`, `internal error system`)
    await fetchNotifyUser([friendId], 'friendRemoved', {id: id})
}