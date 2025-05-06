
export const controller = new AbortController();

const parseSSEMessage  = (raw: string): {event: string, stringData: string} => {
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
        const token = localStorage.getItem('token')
        const res = await fetch(`/user-management/sse`, {
            method: 'GET',
            headers: {
                'Content-Type': 'text/event-stream',
                'Authorization': `Bearer ${token}`
            },
            signal: controller.signal
        })
        if (res.status === 100)
            return;
        if (!res.ok) {
            const error = res.json()
            console.log(error)
            //notifyhandling
            return ;
        }
        console.log('sse connection')
        const reader = res.body?.pipeThrough(new TextDecoderStream()).getReader() ?? null;
        while (reader) {
            const {value, done} = await reader.read();
            if (done) break;
            if (value.startsWith('retry: ')) continue;
            const parse = parseSSEMessage(value)
            if (parse.event in mapEvent)
                mapEvent[parse.event](JSON.parse(parse.stringData));
        }
    } catch (err) {
        console.error(err);
    }
}


const notifyNewFriend = ({id}: { id: number }) => {
    console.log(`this id ${id} have to be add in the friendlist`)

}

const notifyFriendInvitation = ({id}: {id: number}) => {
    console.log(`this friend have to be add in the invitation list`)
}

const notifyFriendRemoved = ({id}: { id: number }) => {
    console.log(`this friend have to be supressed from the friendlist`)
}

const notifyDisconnection = ({id}: { id: number }) => {
    console.log(`this friend have to be marked as unconnected`)
}

const notifyConnection = ({id}: { id: number }) => {
    console.log(`this friend have to be marked as connected`)
}

const notifyJoinTournament = ({id}: { id: number }) => {
    console.log(`the user with the id ${id} joined my tournament`)
}

const notifyQuitTournament = ({id}: { id: number }) => {
    console.log(`the user with the id ${id} Quit my tournament`)
}


const mapEvent : {[key: string] : (data: any) => void} = {
    "joinTournament" : notifyJoinTournament,
    "quitTournament" : notifyQuitTournament,
    "newFriend" : notifyNewFriend,
    "friendInvitation" : notifyFriendInvitation,
    "friendRemoved": notifyFriendRemoved,
    "connection" : notifyConnection,
    "disconnection" : notifyDisconnection,

}