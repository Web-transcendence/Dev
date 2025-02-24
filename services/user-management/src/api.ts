import fastify, { FastifyRequest} from 'fastify'

interface IParam {
    id: string;
}

const app = fastify();

app.get('/user-management/:id', async (req: FastifyRequest<{ Params: IParam }>) => {
    return {
        id: req.params.id,
        name: "testname"
    };
});

app.listen({port: 8000}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on  ${adrr}`)
})