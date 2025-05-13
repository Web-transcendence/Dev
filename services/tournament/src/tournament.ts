import {INTERNAL_PASSWORD, tournamentSessions} from "./api.js"
import {ConflictError, ServerError} from "./error.js";
import {fetchNotifyUser} from "./utils.js";


async function fetchMatch(id1: number, id2: number) {
    const response = await fetch(`http://match-server:4443/tournamentGame`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'authorization': `${INTERNAL_PASSWORD}`
        },
        body: JSON.stringify({id1: id1, id2: id2})
    })
    if (response.ok) {
        const winner = await response.json()
        return winner.id
    }
    throw new Error(`CREATE A REAL ERROR MESSAGE`)
}

export class tournament {

    private participantId: number[]
    private status: 'waiting' | 'started'
    private result
    maxPlayer: number

    constructor(maxPlayer: number) {
        if ([4, 8, 16, 32].includes(maxPlayer))
            this.maxPlayer = maxPlayer
        else
            throw new ServerError(`cannot create a tournament with ${maxPlayer} player`, 500)
        this.participantId = []
        this.status = 'waiting'
        this.result = {}
    }

    hasParticipant(userId: number): boolean {
        return this.participantId.includes(userId)
    }

    async addParticipant(participantId: number) {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`, `This tournament is already started`)
        if (this.participantId.length >= this.maxPlayer)
            throw new ConflictError(`maxPlayer has already is reached`, `This tournament is full`)

        for (const [id, tournament] of tournamentSessions)
            if (tournament.hasParticipant(participantId))
                throw new ConflictError(`this user has already another tournament`, `you cannot participate to multiple tournament`)

        await fetchNotifyUser(this.participantId, 'joinTournament', {id: participantId, maxPlayer: this.maxPlayer})
        this.participantId.push(participantId)
    }

    getData(): {participants: number[], maxPlayer: number, status: string} {
        return {
            participants: this.participantId,
            maxPlayer: this.maxPlayer,
            status: this.status,
        }
    }

    async quit(id: number){
        if (this.status === 'started') {
            return ;
        }
        else {
            this.participantId = this.participantId.filter(participantId => participantId !== id)
            await fetchNotifyUser(this.participantId, 'quitTournament', {id: id, maxPlayer: this.maxPlayer})
        }
    }



    async bracketHandler(bracket: number[]): Promise<number> {
        if (bracket.length === 2) {
            //await startMatch(bracket[0], bracket[1])
            const winnerId = await fetchMatch(bracket[0], bracket[1])
            console.log(`Match terminé: ${bracket[0]} vs ${bracket[1]}`)
            return winnerId
        }
        else {
            const winner = await Promise.all([
                this.bracketHandler(bracket.slice(0, bracket.length / 2)),
                this.bracketHandler(bracket.slice(bracket.length / 2))
            ])

            //await startMatch(bracket[0], bracket[1])
            const winnerId = await fetchMatch(winner[0], winner[1])
            console.log(`Match terminé: ${winner[0]} vs ${winner[1]}`)
            return winnerId
        }
    }


    async launch(): Promise<string> {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`, `cannot launch this tournament`)
        if (this.participantId.length !== this.maxPlayer)
            throw new ConflictError(`participant have to be 4, 8, 16 or 32`, `cannot launch this tournament without ${this.maxPlayer} participant`)

        const arr = [...this.participantId]
        for (let i = arr.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [arr[i], arr[j]] = [arr[j], arr[i]]
        }

        const winner = await this.bracketHandler(arr)
        console.log(`${winner} is the winner`)

        return winner.toString()
    }
}


