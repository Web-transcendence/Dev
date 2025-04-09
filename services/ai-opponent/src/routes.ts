import { FastifyReply, FastifyRequest, FastifyInstance } from "fastify";





export default async function AiRoutes(app: FastifyInstance) {
	app.get('/call', (req: FastifyRequest, res: FastifyReply) => {
		try{
			console.log(req.headers);
			return res.status(200).send({success: 'yes'})
		} catch (err) {
			console.log(err);
		}
	})
}
