import {tournamentSessions} from "./api.js"
import {ConflictError, ServerError} from "./error.js";

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

    addParticipant(participantId: number) {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`, `This tournament is already started`)
        if (this.participantId.length >= this.maxPlayer)
            throw new ConflictError(`maxPlayer has already is reached`, `This tournament is full`)

        for (const [id, tournament] of tournamentSessions)
            if (tournament.hasParticipant(participantId))
                throw new ConflictError(`this user has already another tournament`, `internal error system`)

        this.participantId.push(participantId)
    }

    getData(): {participants: number[], maxPlayer: number, status: string} {
        return {
            participants: this.participantId,
            maxPlayer: this.maxPlayer,
            status: this.status,
        }
    }

    quit(id: number): void {
        this.participantId.filter(participantId => participantId !== id)
    }

    async bracketHandler(bracket: number[]): Promise<number> {
        if (bracket.length === 2) {
            //await startMatch(bracket[0], bracket[1])
            console.log(`Match entre ${bracket[0]} et ${bracket[1]}`)
            await new Promise(res => setTimeout(res, 1000))
            console.log(`Match terminé: ${bracket[0]} vs ${bracket[1]}`)
            return bracket[Math.floor(Math.random() > 0.5 ? 0 : 1)]
        }
        else {
            const winner = await Promise.all([
                this.bracketHandler(bracket.slice(0, bracket.length / 2)),
                this.bracketHandler(bracket.slice(bracket.length / 2))
            ])

            //await startMatch(bracket[0], bracket[1])
            console.log(`Match entre ${winner[0]} et ${winner[1]}`)
            await new Promise(res => setTimeout(res, 1000))
            console.log(`Match terminé: ${winner[0]} vs ${winner[1]}`)
            return bracket[Math.floor(Math.random() > 0.5 ? 0 : 1)]
        }
    }

    async launch(): Promise<string> {
        if (this.status === 'started')
            throw new ConflictError(`this tournament has already started`, `internal error system`)
        if (this.participantId.length !== this.maxPlayer)
            throw new ConflictError(`participant have to be 4, 8, 16 or 32`, `internal error system`)

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


