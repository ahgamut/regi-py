/* Embind bindings for the regicide core engine -- the WASM analogue of the
 * pybind11 exports.cc. JavaScript owns the game loop and drives the engine one
 * decision at a time: it subclasses Strategy (the offered combos arrive as an
 * argument to the attack/defense callbacks) and steps GameState to completion. */
#include <emscripten/bind.h>
#include <regi.h>
#include <dfsel.h>
#include <phaseinfo.h>
#include <rng.h>
#include <string>
#include <vector>

using namespace emscripten;
using namespace regi;

/* No-op log so a GameState can run without ConsoleLog's stdout I/O. */
struct NoOpLog : public BaseLog
{
    void attack(const Player &, const Enemy &, const Combo &, const i32,
                const GameState &) override {}
    void defend(const Player &, const Combo &, const i32, const GameState &) override {}
    void redirect(const Player &, const i32, const GameState &) override {}
    void failBlock(const Player &, const i32, const i32, const GameState &) override {}
    void fullBlock(const Player &, const i32, const i32, const GameState &) override {}
    void drawOne(const Player &) override {}
    void cannotDrawDeckEmpty(const Player &, const GameState &) override {}
    void replenish(const i32) override {}
    void enemyKill(const Enemy &, const GameState &) override {}
    void state(const GameState &) override {}
    void debug(const GameState &) override {}
    void startgame(const GameState &) override {}
    void endgame(EndGameReason, const GameState &) override {}
    void postgame(const GameState &) override {}
};

/* JS-subclassable Strategy: JS implements the four decision methods. */
struct StrategyWrapper : public wrapper<Strategy>
{
    EMSCRIPTEN_WRAPPER(StrategyWrapper);
    i32 setup(const Player &p, const GameState &g) override
    {
        return call<i32>("setup", p, g);
    }
    i32 getAttackIndex(const std::vector<Combo> &combos, const Player &p, bool yieldAllowed,
                       const GameState &g) override
    {
        return call<i32>("getAttackIndex", combos, p, yieldAllowed, g);
    }
    i32 getDefenseIndex(const std::vector<Combo> &combos, const Player &p, i32 damage,
                        const GameState &g) override
    {
        return call<i32>("getDefenseIndex", combos, p, damage, g);
    }
    i32 getRedirectIndex(const Player &p, const GameState &g) override
    {
        return call<i32>("getRedirectIndex", p, g);
    }
};

/* GameState lifecycle helpers (mirror exports.cc's gs* free functions). */
static GameStatus gsInitialize(GameState &g)
{
    g.init();
    g.setup();
    return g.status;
}
static GameStatus gsInitString(GameState &g, std::string s)
{
    PhaseInfo info;
    loadPhaseInfoOrFail(info, s);
    g.initPhaseInfo(info);
    g.setup();
    return g.status;
}
static GameStatus gsInitPhaseInfo(GameState &g, PhaseInfo &info)
{
    g.initPhaseInfo(info);
    g.setup();
    return g.status;
}
static std::string gsExportString(GameState &g)
{
    PhaseInfo info;
    g.loadPhaseInfoForExport(info);
    return info.toString();
}
static PhaseInfo gsExportPhaseInfo(GameState &g)
{
    PhaseInfo info;
    g.loadPhaseInfoForExport(info);
    return info;
}

static PhaseInfo phaseFromString(std::string s)
{
    PhaseInfo info;
    loadPhaseInfoOrFail(info, s);
    return info;
}
static PhaseInfo phaseRandomizeFrom(const PhaseInfo &other, i32 currentID)
{
    PhaseInfo info(other);
    info.randomize(currentID);
    return info;
}

EMSCRIPTEN_BINDINGS(regicore)
{
    enum_<GameStatus>("GameStatus")
        .value("LOADING", GameStatus::LOADING)
        .value("RUNNING", GameStatus::RUNNING)
        .value("ENDED", GameStatus::ENDED);

    register_vector<Card>("VectorCard");
    register_vector<Enemy>("VectorEnemy");
    register_vector<Combo>("VectorCombo");
    register_vector<std::vector<Card>>("VectorVectorCard");

    class_<Card>("Card")
        .property("entry", +[](const Card &c) { return (int)c.entry(); })
        .property("suit", +[](const Card &c) { return (int)c.suit(); })
        .property("strength", &Card::strength)
        .property("location", &Card::toLocation)
        .property("is_yield", &Card::isYield);

    class_<Enemy, base<Card>>("Enemy").property("hp", &Enemy::hp);

    class_<Combo>("Combo")
        .property("parts", &Combo::parts)
        .property("base_damage", &Combo::getBaseDamage)
        .property("base_defense", &Combo::getBaseDefense)
        .property("can_attack", +[](const Combo &c) { return c.valid(true) != 0; });

    class_<Player>("Player")
        .property("id", &Player::id)
        .property("alive", &Player::alive)
        .property("cards", &Player::cards);

    class_<PhaseInfo>("PhaseInfo")
        .class_function("from_string", &phaseFromString)
        .class_function("randomize_from", &phaseRandomizeFrom)
        .function("to_string", &PhaseInfo::toString)
        .function("combo_damage", &PhaseInfo::comboDamage)
        .function("combo_block", &PhaseInfo::comboBlock)
        .function("current_block", &PhaseInfo::currentBlock)
        .property("num_players", &PhaseInfo::numPlayers)
        .property("game_endvalue", &PhaseInfo::gameHasEnded)
        .property("active_player", &PhaseInfo::activePlayerID)
        .property("phase_attacking", &PhaseInfo::currentPhaseIsAttack)
        .property("player_cards", &PhaseInfo::player_cards)
        .property("draw_pile", &PhaseInfo::drawPile)
        .property("enemy_pile", &PhaseInfo::enemyPile)
        .property("used_combos", &PhaseInfo::usedPile);

    class_<BaseLog>("BaseLog");
    class_<NoOpLog, base<BaseLog>>("NoOpLog").constructor<>();

    class_<Strategy>("Strategy")
        .allow_subclass<StrategyWrapper>("StrategyWrapper")
        .function("setup", &Strategy::setup, pure_virtual())
        .function("getAttackIndex", &Strategy::getAttackIndex, pure_virtual())
        .function("getDefenseIndex", &Strategy::getDefenseIndex, pure_virtual())
        .function("getRedirectIndex", &Strategy::getRedirectIndex, pure_virtual());
    class_<RandomStrategy, base<Strategy>>("RandomStrategy").constructor<>();
    class_<DamageStrategy, base<Strategy>>("DamageStrategy").constructor<>();

    class_<GameState>("GameState")
        .constructor<BaseLog &>()
        .function("add_player", &GameState::addPlayer)
        .function("initialize", &gsInitialize)
        .function("init_string", &gsInitString)
        .function("init_phaseinfo", &gsInitPhaseInfo)
        .function("export_string", &gsExportString)
        .function("export_phaseinfo", &gsExportPhaseInfo)
        .function("is_runnable", &GameState::gameRunning)
        .function("step", &GameState::onePhase)
        .function("num_players", &GameState::totalPlayers)
        .property("status", &GameState::status)
        .property("active_player", &GameState::activePlayerID)
        .property("phase_attacking", &GameState::currentPhaseIsAttack)
        .property("phase_count", &GameState::phaseCount)
        .property("draw_pile", &GameState::drawPile)
        .property("enemy_pile", &GameState::enemyPile)
        .property("used_combos", &GameState::usedPile);

    function("seed", +[](unsigned v) { regi::seed((std::uint64_t)v); });
}
