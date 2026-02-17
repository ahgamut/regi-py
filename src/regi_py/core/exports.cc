#include <regi.h>
#include <console.h>
#include <dfsel.h>
#include <phaseinfo.h>
//
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/native_enum.h>
namespace py = pybind11;
using namespace regi;

template <typename T>
std::string stringify(const T &t)
{
    std::ostringstream ss;
    ss << t;
    return ss.str();
}

void bind_enums(pybind11::object &m)
{
    py::native_enum<Suit>(m, "Suit", "enum.IntEnum")
        .value("GLITCH", Suit::GLITCH)
        .value("CLUBS", Suit::CLUBS)
        .value("DIAMONDS", Suit::DIAMONDS)
        .value("HEARTS", Suit::HEARTS)
        .value("SPADES", Suit::SPADES)
        .export_values()
        .finalize();

    py::native_enum<Entry>(m, "Entry", "enum.IntEnum")
        .value("JOKER", Entry::JOKER)
        .value("ACE", Entry::ACE)
        .value("TWO", Entry::TWO)
        .value("THREE", Entry::THREE)
        .value("FOUR", Entry::FOUR)
        .value("FIVE", Entry::FIVE)
        .value("SIX", Entry::SIX)
        .value("SEVEN", Entry::SEVEN)
        .value("EIGHT", Entry::EIGHT)
        .value("NINE", Entry::NINE)
        .value("TEN", Entry::TEN)
        .value("JACK", Entry::JACK)
        .value("QUEEN", Entry::QUEEN)
        .value("KING", Entry::KING)
        .export_values()
        .finalize();

    py::native_enum<Powers>(m, "SuitPower", "enum.Flag")
        .value("DOUBLE_DAMAGE", Powers::CLUBS_DOUBLE)
        .value("DRAW_CARDS", Powers::DIAMONDS_DRAW)
        .value("REPLENISH", Powers::HEARTS_REPLENISH)
        .value("BLOCK", Powers::SPADES_BLOCK)
        .value("NERF", Powers::JOKER_NERF)
        .export_values()
        .finalize();

    py::native_enum<GameStatus>(m, "GameStatus", "enum.IntEnum")
        .value("LOADING", GameStatus::LOADING)
        .value("RUNNING", GameStatus::RUNNING)
        .value("ENDED", GameStatus::ENDED)
        .export_values()
        .finalize();

    py::native_enum<EndGameReason>(m, "EndGameReason", "enum.IntEnum")
        .value("INVALID_START_PLAYER_COUNT", EndGameReason::INVALID_START_PLAYER_COUNT)
        .value("INVALID_START_PLAYER_SETUP", EndGameReason::INVALID_START_PLAYER_SETUP)
        .value("NO_ENEMIES", EndGameReason::NO_ENEMIES)
        .value("BLOCK_FAILED", EndGameReason::BLOCK_FAILED)
        .value("ATTACK_FAILED", EndGameReason::ATTACK_FAILED)
        .value("REDIRECT_FAILED", EndGameReason::REDIRECT_FAILED)
        .value("PLAYER_DEAD", EndGameReason::PLAYER_DEAD)
        .export_values()
        .finalize();
}

void bind_cards(pybind11::object &m)
{
    py::class_<Card>(m, "Card")
        .def_property_readonly("entry", &Card::entry)
        .def_property_readonly("suit", &Card::suit)
        .def_property_readonly("strength", &Card::strength)
        .def_property_readonly("index", &Card::toIndex)
        .def(
            "__eq__", [](const Card &c1, const Card &c2) { return c1 == c2; },
            py::is_operator())
        .def(
            "__lt__", [](const Card &c1, const Card &c2) { return c1 < c2; },
            py::is_operator())
        .def(
            "__gt__", [](const Card &c1, const Card &c2) { return c1 > c2; },
            py::is_operator())
        .def("__hash__",
             [](const Card &c) { return std::hash<std::string>{}(stringify<Card>(c)); })
        .def("__repr__", &stringify<Card>)
        .def("__str__", &stringify<Card>);

    py::class_<Enemy>(m, "Enemy")
        .def_readonly("hp", &Enemy::hp)
        .def_property_readonly("entry", &Enemy::entry)
        .def_property_readonly("suit", &Enemy::suit)
        .def_property_readonly("strength", &Enemy::strength)
        .def_property_readonly("index", &Enemy::toIndex)
        .def("__repr__", &stringify<Enemy>)
        .def("__str__", &stringify<Enemy>);

    py::class_<Combo>(m, "Combo")
        .def_readonly("parts", &Combo::parts)
        .def_property_readonly("can_attack",
                               [](Combo &c) { return c.valid(true) != 0; })
        .def_property_readonly("base_damage", &Combo::getBaseDamage)
        .def_property_readonly("base_defense", &Combo::getBaseDefense)
        .def_property_readonly("bitwise", &Combo::getBitrep)
        .def("__repr__", &stringify<Combo>)
        .def("__str__", &stringify<Combo>);
}

class PyBaseStrategy : public Strategy, py::trampoline_self_life_support
{
    /* TODO: Is there no other way? */
   public:
    using Strategy::Strategy;
    i32 setup(const Player &player, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(i32, Strategy, setup, player, g);
    }
    i32 getDefenseIndex(const std::vector<Combo> &combos, const Player &player,
                        i32 damage, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(i32, Strategy, getDefenseIndex, combos, player, damage,
                               g);
    }
    i32 getAttackIndex(const std::vector<Combo> &combos, const Player &player,
                       bool yieldAllowed, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(i32, Strategy, getAttackIndex, combos, player,
                               yieldAllowed, g);
    }
    i32 getRedirectIndex(const Player &player, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(i32, Strategy, getRedirectIndex, player, g);
    }
};

class PyRandomStrategy : public RandomStrategy, py::trampoline_self_life_support
{
    using RandomStrategy::RandomStrategy;
};

class PyDamageStrategy : public DamageStrategy, py::trampoline_self_life_support
{
    using DamageStrategy::DamageStrategy;
};

void bind_strat(pybind11::object &m)
{
    py::class_<Strategy, PyBaseStrategy, py::smart_holder> base(m, "BaseStrategy");
    base.def(py::init<>())
        .def("setup", &Strategy::setup)
        .def("getAttackIndex", &Strategy::getAttackIndex)
        .def("getRedirectIndex", &Strategy::getRedirectIndex)
        .def("getDefenseIndex", &Strategy::getDefenseIndex);
    py::class_<RandomStrategy, PyRandomStrategy, py::smart_holder>(m, "RandomStrategy",
                                                                   base)
        .def(py::init<>())
        .def_property_readonly_static("__strat_name__",
                                      [](py::object self)
                                      {
                                          (void)self;
                                          return "random";
                                      })
        .def("setup", &RandomStrategy::setup)
        .def("getAttackIndex", &RandomStrategy::getAttackIndex)
        .def("getRedirectIndex", &RandomStrategy::getRedirectIndex)
        .def("getDefenseIndex", &RandomStrategy::getDefenseIndex);
    py::class_<DamageStrategy, PyDamageStrategy, py::smart_holder>(m, "DamageStrategy",
                                                                   base)
        .def(py::init<>())
        .def_property_readonly_static("__strat_name__",
                                      [](py::object self)
                                      {
                                          (void)self;
                                          return "damage";
                                      })
        .def("setup", &DamageStrategy::setup)
        .def("getAttackIndex", &DamageStrategy::getAttackIndex)
        .def("getRedirectIndex", &DamageStrategy::getRedirectIndex)
        .def("getDefenseIndex", &DamageStrategy::getDefenseIndex);
}

void bind_player(pybind11::object &m)
{
    py::class_<Player>(m, "Player")
        .def_readonly("cards", &Player::cards)
        .def_readonly("id", &Player::id)
        .def_readonly("alive", &Player::alive)
        .def_property_readonly("strategy",
                               [](Player &player)
                               {
                                   py::object strat = py::cast(player.strat);
                                   return py::getattr(strat, "__strat_name__");
                               })
        .def("__repr__", &stringify<Player>)
        .def("__str__", &stringify<Player>);
}

class PyBaseLog : public BaseLog, py::trampoline_self_life_support
{
    /* TODO: Is there no other way? */
   public:
    using BaseLog::BaseLog;
    void attack(const Player &player, const Enemy &enemy, const Combo &combo,
                const i32 damage, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, attack, player, enemy, combo, damage, g);
    }
    void defend(const Player &player, const Combo &combo, const i32 damage,
                const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, defend, player, combo, damage, g);
    }
    void redirect(const Player &player, const i32 nextPlayerID,
                  const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, redirect, player, nextPlayerID, g);
    }
    void failBlock(const Player &player, const i32 damage, const i32 maxblock,
                   const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, failBlock, player, damage, maxblock, g);
    }
    void fullBlock(const Player &player, const i32 damage, const i32 fblock,
                   const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, fullBlock, player, damage, fblock, g);
    }
    void drawOne(const Player &player) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, drawOne, player);
    }
    void cannotDrawDeckEmpty(const Player &player, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, cannotDrawDeckEmpty, player, g);
    }
    void replenish(const i32 n) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, replenish, n);
    }
    void enemyKill(const Enemy &enemy, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, enemyKill, enemy, g);
    }
    void state(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, state, g);
    }
    void debug(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, debug, g);
    }
    void startgame(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, startgame, g);
    }
    void endgame(EndGameReason e, const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, endgame, e, g);
    }
    void postgame(const GameState &g) override
    {
        PYBIND11_OVERRIDE_PURE(void, BaseLog, postgame, g);
    }
};

class PyConsoleLog : public ConsoleLog, py::trampoline_self_life_support
{
    using ConsoleLog::ConsoleLog;
};

void bind_log(pybind11::object &m)
{
    py::class_<BaseLog, PyBaseLog, py::smart_holder> base(m, "BaseLog");
    base.def(py::init<>())
        .def("attack", &BaseLog::attack)
        .def("defend", &BaseLog::defend)
        .def("redirect", &BaseLog::redirect)
        .def("failBlock", &BaseLog::failBlock)
        .def("fullBlock", &BaseLog::fullBlock)
        .def("drawOne", &BaseLog::drawOne)
        .def("cannotDrawDeckEmpty", &BaseLog::cannotDrawDeckEmpty)
        .def("replenish", &BaseLog::replenish)
        .def("enemyKill", &BaseLog::enemyKill)
        .def("state", &BaseLog::state)
        .def("debug", &BaseLog::debug)
        .def("startgame", &BaseLog::startgame)
        .def("endgame", &BaseLog::endgame)
        .def("postgame", &BaseLog::postgame);
    py::class_<ConsoleLog, PyConsoleLog, py::smart_holder>(m, "CXXConsoleLog", base)
        .def(py::init<>());
}

void loadPhaseInfoOrFail(PhaseInfo &info, std::string s)
{
    if (!info.loadFromString(s))
    {
        throw std::runtime_error("unable to load info from string: " + s);
    }
}

void bind_phaseinfo(pybind11::object &m)
{
    py::class_<PhaseInfo>(m, "PhaseInfo")
        .def_static("from_string",
                    [](std::string s)
                    {
                        PhaseInfo info;
                        loadPhaseInfoOrFail(info, s);
                        return info;
                    })
        .def_readonly("num_players", &PhaseInfo::numPlayers)
        .def_readonly("game_endvalue", &PhaseInfo::gameHasEnded)
        .def_readonly("active_player", &PhaseInfo::activePlayerID)
        .def_readonly("phase_attacking", &PhaseInfo::currentPhaseIsAttack)
        .def_readonly("past_yields", &PhaseInfo::pastYieldsInARow)
        .def_readonly("player_cards", &PhaseInfo::player_cards)
        .def_readonly("draw_pile", &PhaseInfo::drawPile)
        .def_readonly("discard_pile", &PhaseInfo::discardPile)
        .def_readonly("enemy_pile", &PhaseInfo::enemyPile)
        .def_readonly("used_combos", &PhaseInfo::usedPile)
        .def("__hash__",
             [](const PhaseInfo &info) { return std::hash<std::string>{}(info.toString()); })
        .def("to_string", &PhaseInfo::toString)
        .def("__str__", &PhaseInfo::toString)
        .def("__repr__", &PhaseInfo::toString)
        .def(py::pickle(
            [](const PhaseInfo& info) { // dump
                return py::make_tuple(info.toString());
            },
            [](py::tuple t) { // load
                std::string s0 = t[0].cast<std::string>();
                PhaseInfo info;
                loadPhaseInfoOrFail(info, s0);
                return info;
            }
        ));
}

void bind_gamestate(pybind11::object &m)
{
    py::class_<GameState>(m, "GameState")
        .def(py::init([](BaseLog &log) { return GameState(log); }),
             py::keep_alive<1, 2>())
        .def("add_player", &GameState::addPlayer, py::keep_alive<1, 2>())
        .def_property_readonly("num_players", &GameState::totalPlayers)
        .def_property_readonly("hand_size", &GameState::getHandSize)
        .def_readonly("active_player", &GameState::activePlayerID)
        .def_readonly("phase_attacking", &GameState::currentPhaseIsAttack)
        .def_readonly("phase_count", &GameState::phaseCount)
        .def_readonly("past_yields", &GameState::pastYieldsInARow)
        .def_readonly("status", &GameState::status)
        .def_readonly("players", &GameState::players)
        .def_readonly("draw_pile", &GameState::drawPile)
        .def_readonly("discard_pile", &GameState::discardPile)
        .def_readonly("enemy_pile", &GameState::enemyPile)
        .def_readonly("used_combos", &GameState::usedPile)
        .def("get_current_block", [](GameState &g, Enemy &e) { return g.calcBlock(e); })
        .def("get_combo_damage", &GameState::calcDamageOfCombo)
        .def("get_combo_block", &GameState::calcBlockOfCombo)
        .def("start_loop", &GameState::startLoop)
        .def("_step", &GameState::onePhase)
        .def("_set_status", [](GameState &g, GameStatus s) { g.status = s; })
        .def_property_readonly("is_runnable", &GameState::gameRunning)
        .def("initialize",
             [](GameState &g)
             {
                 g.init();
                 g.setup();
                 return g.status;
             })
        .def("_init_random",
             [](GameState &g)
             {
                 g.initRandom();
                 g.setup();
                 return g.status;
             })
        .def("_init_phaseinfo",
             [](GameState &g, PhaseInfo &info)
             {
                 g.initPhaseInfo(info);
                 g.setup();
                 return g.status;
             })
        .def("export_phaseinfo",
             [](GameState &g)
             {
                 PhaseInfo info;
                 g.loadPhaseInfoForExport(info);
                 return info;
             })
        .def("_init_string",
             [](GameState &g, std::string s)
             {
                 PhaseInfo info;
                 loadPhaseInfoOrFail(info, s);
                 g.initPhaseInfo(info);
                 g.setup();
                 return g.status;
             })
        .def("export_string",
             [](GameState &g)
             {
                 PhaseInfo info;
                 g.loadPhaseInfoForExport(info);
                 return info.toString();
             });
}

PYBIND11_MODULE(core, m)
{
    m.doc() = "c++ module for regicide game mechanics";
    bind_enums(m);
    bind_cards(m);
    bind_strat(m);
    bind_player(m);
    bind_log(m);
    bind_phaseinfo(m);
    bind_gamestate(m);
}
